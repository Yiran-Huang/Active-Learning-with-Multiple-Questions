import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
import random
import time
import copy
import torch.nn.functional as F
from get_model import *
from functions import *
from for_badge import *

#### Define cross-entropy loss ####
def entropy_loss(proby, prob):
    probyonhot = torch.zeros(prob.size(0), prob.size(1)).to(prob.device)
    probyonhot.scatter_(1, proby.unsqueeze(1), 1)
    en = torch.sum(probyonhot * (-torch.log(prob + 1e-5)), dim=1)
    return torch.sum(en)

def entropy_lossonehot(proby, prob):
    en = torch.sum(proby * (-torch.log(prob + 1e-5)), dim=1)
    return torch.sum(en)

#### get predicting result ####
def get_pred_result(model, device, x, y, L,batch_size=512,
                    needprob=False, needlogits=False,trans_x=nn.Identity(),showprocess=False,need_embed=False):
    dataloadertemp = DataLoader(CustomDataSet(x, y), batch_size=batch_size, shuffle=False)
    logitall = torch.empty((0, L))
    emball=torch.empty((0,model.get_embedding_dim()))
    with torch.no_grad():
        dd = 0
        for xx, yy in dataloadertemp:
            dd += 1
            if showprocess:
                print(f'\rProgress: |{"█" * int(50 * dd / len(dataloadertemp))}{"-" * (50 - int(50 * dd / len(dataloadertemp)))}| {dd / len(dataloadertemp) * 100:.2f}% Complete',
                    end='')
            xx, yy = trans_x(xx,device).to(device), yy.to(device)
            model.eval()
            if need_embed:
                logitthis,emb = model(xx,need_embed=need_embed)
                logitthis=logitthis.cpu()
                emb=emb.cpu()
                emball=torch.cat((emball,emb),0)
            else:
                logitthis = model(xx).cpu()
            logitall = torch.cat((logitall, logitthis), 0)
    acc = torch.sum(torch.argmax(logitall, 1) == y).float() / len(y)
    proball = F.softmax(logitall,1)
    loss = entropy_loss(y, proball) / proball.shape[0]
    entropy = torch.mean(-torch.sum(proball * torch.log(proball + 1e-5), 1))
    variance = torch.mean(torch.sum(proball * (1-proball), 1))
    least_conf = torch.mean(1-torch.max(proball, 1)[0])
    if need_embed:
        return torch.tensor([acc, loss, entropy,variance,least_conf]), logitall, emball
    if needprob:
        return torch.tensor([acc, loss, entropy,variance,least_conf]), proball
    elif needlogits:
        return torch.tensor([acc, loss, entropy,variance,least_conf]), logitall
    else:
        return torch.tensor([acc, loss, entropy,variance,least_conf])

#### distance function ####
def dist_a2B(a, B, p=2, device="cpu"):
    a = a.to(device)
    B = B.to(device)
    diff = torch.abs(B - a)
    dims = [d for d in range(B.dim()) if d != 0]
    output = (torch.sum(diff ** p, dims)) ** (1 / p)
    output.cpu()
    return output

def dist_A2B_p2(A, B, device="cpu",batch_size=2048):
    A = A.view(A.size(0), -1)
    B = B.view(B.size(0), -1)
    Aloader = DataLoader(CustomDatasetx(A), batch_size=batch_size, shuffle=False)
    Bloader = DataLoader(CustomDatasetx(B), batch_size=batch_size, shuffle=False)
    disall=torch.empty((0,B.shape[0]))
    for aa in Aloader:
        distemp=torch.empty((aa.shape[0],0))
        aa=aa.to(device)
        for bb in Bloader:
            bb=bb.to(device)
            disab=(torch.sum(aa ** 2, 1).reshape(aa.shape[0], 1) + torch.sum(bb ** 2, 1).reshape(1, bb.shape[0]) - 2 * (
                aa @ bb.t())) ** 0.5
            disab=disab.cpu()
            distemp=torch.cat((distemp,disab),1)
        disall=torch.cat((disall,distemp),0)
    del aa,bb,disab,distemp
    disall[torch.isnan(disall)] = 0
    return disall

def dist_subset_p2(loc, B, loc_before_B=True, device="cpu",batch_size=2048):
    A = B[loc]
    disAB=dist_A2B_p2(A,B,device=device,batch_size=batch_size)
    if loc_before_B:
        return disAB
    else:
        return disAB.t()


def build_model_A(model,device,y, x, maxm,initial_model=None,lr=None,
                  question_set=[None], question_and_answer=None, hyper_regular=0,
                  epochs=50,trans_x=nn.Identity(), return_loss=False, not_optimize=False, batch_size=512,
                  layers_to_regularize=None,break_num=5,return_full_loss=False):
    #the first len(y) x are corresponding to y!
    #the rest of x are in the order according to locations in question_and_answer rowwisely!
    #question_and_answer has 3+max(m) columns, the first is question number,
    #the second is answer, the third is the class inside question
    # the next m are locations of corresponding x.
    #the rest max(m)-m are filled by -1
    if break_num is None:
        break_num=epochs+1
    l1penalty = hyper_regular
    if lr is None:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    model = model.to(device)
    if question_and_answer is None:
        question_and_answer=torch.empty((0,maxm+3),dtype=torch.int64)

    #check any initial parameters, and should we optimize or not
    if initial_model is not None:
        model.load_state_dict(initial_model)
    if not_optimize:
        return

    #get q_and_a
    q_and_a=question_and_answer.clone()
    ny=len(y)
    temp1dvec = q_and_a[:, 3:].clone().reshape(-1)
    loctemp1d = torch.where(temp1dvec != -1)[0]
    temp1dvec[loctemp1d] = torch.arange(len(loctemp1d), dtype=torch.int64)+ny
    q_and_a[:, 3:] = temp1dvec.reshape(q_and_a.shape[0], q_and_a.shape[1] - 3)
    datalen=ny+q_and_a.shape[0]
    #get_loss_f
    class loss_f_ALMQ(nn.Module):
        def __init__(self):
            super(loss_f_ALMQ, self).__init__()

        def forward(self, probpre,y,q_and_athis):
            lossterm1 = entropy_loss(y, probpre)
            losstermq = 0
            for i in range(q_and_athis.shape[0]):
                numnow = q_and_athis[i, 0]
                anow = q_and_athis[i, 1]
                cnow = q_and_athis[i,2]
                locnow = q_and_athis[i, 3:]
                locnow = locnow[locnow!=-1]
                losstermq += question_set[numnow][0](anow, torch.tensor([0,cnow],dtype=torch.int64), probpre[locnow])
            loss_now = (lossterm1 + losstermq) / (len(y) + probpre.shape[0])
            return loss_now
    custom_loss = loss_f_ALMQ().to(device)

    #get batch size as well as loc for each batch
    bs_true = get_bs(batch_size,datalen)
    numbers = np.arange(0, datalen)
    random.shuffle(numbers)
    numbers = torch.tensor(numbers,dtype=torch.int64)
    loc_each_batch=[]
    for i in range(np.array(np.ceil((datalen)/(bs_true)),dtype="int64")):
        locmin=bs_true*i
        locmax=min(locmin+bs_true,datalen)
        loc_each_batch.append(numbers[torch.arange(locmin,locmax,1)])

    losses = []
    opt_loss=torch.tensor(100000.0)
    opt_times_count=0
    for epoch in range(epochs):
        ncount=0
        loss_this_epoch=0
        for batchnumber in range(len(loc_each_batch)):
            #load data for this batch
            loc_this_batch , _ = torch.sort(loc_each_batch[batchnumber])
            ny_this_batch = len(loc_this_batch[loc_this_batch<ny])
            y_this_batch = y[loc_this_batch[:ny_this_batch]].to(device)
            q_and_a_this_batch = q_and_a[loc_this_batch[ny_this_batch:]-ny].clone()
            temp1dvec = q_and_a_this_batch[:, 3:].clone().reshape(-1)
            loctemp1d = torch.where(temp1dvec != -1)[0]
            loc_x_this_batch = torch.cat((loc_this_batch[:ny_this_batch],temp1dvec[loctemp1d]))
            x_this_batch = trans_x(x[loc_x_this_batch],device).to(device)
            temp1dvec[loctemp1d] = torch.arange(len(loctemp1d), dtype=torch.int64) + ny_this_batch
            q_and_a_this_batch[:, 3:] = temp1dvec.reshape(q_and_a_this_batch.shape[0], q_and_a_this_batch.shape[1] - 3)
            q_and_a_this_batch = q_and_a_this_batch.to(device)


            model.train()
            prob_this_batch = F.softmax(model(x_this_batch),1)
            loss = custom_loss(prob_this_batch,y_this_batch,q_and_a_this_batch)
            l1_norm = 0.0
            if layers_to_regularize is not None:
                for idx, (name, param) in enumerate(model.named_parameters()):
                    if name in layers_to_regularize:
                        l1_norm += torch.norm(param, p=1)
            loss += l1penalty * l1_norm
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #record
            loss_this_batch=loss.item()
            losses.append(loss_this_batch)
            n_this_batch=len(loc_this_batch)
            loss_this_epoch = (loss_this_epoch * ncount + loss_this_batch*n_this_batch)/(ncount+n_this_batch)
            ncount += n_this_batch
        #for early drop
        if loss_this_epoch<opt_loss:
            opt_loss=loss_this_epoch
            opt_times_count = 0
            best_model_para = model.state_dict()
            best_opt_para = optimizer.state_dict()
        else:
            opt_times_count += 1
        if opt_times_count==break_num:
            model.load_state_dict(best_model_para)
            break
    if return_loss:
        if return_full_loss:
            return losses
        else:
            return losses[len(losses) - 1]


#### exploration and exploitation ####
def exploration_and_exploitation(x, lochave, device, explo_rate=0.75, ini_quantile=0.05, threshold_len=6,
                                 return_radius=True):
    nXall = x.shape[0]
    if nXall > 1000:
        loctemp = np.random.choice(range(x.shape[0]), 1000, replace=False)
    else:
        loctemp = np.array(range(nXall))
    threshold_dis = torch.tensor(np.quantile(dist_A2B_p2(x[loctemp], x[loctemp], device=device).numpy(), ini_quantile),dtype=torch.float32)
    threshold_dis = torch.arange(threshold_len - 1, 0 - 1, -1).float() / (threshold_len - 1) * threshold_dis
    threshold_dis.cpu()
    dis_have2all = torch.min(dist_subset_p2(lochave, x, device=device), 0)[0]
    for quan in threshold_dis:
        locconsider = torch.where(dis_have2all > quan)[0]
        locconsider = locconsider.squeeze()
        if (locconsider.dim() > 0):
            lenlocconsider = len(locconsider)
        else:
            lenlocconsider = 1
        if lenlocconsider > (nXall * explo_rate):
            break
    del dis_have2all
    if return_radius:
        return locconsider, quan
    else:
        return locconsider

#### MCdrop ####
def get_probmc(model,x,L,times,batch_infer_size,trans_x,device):
    numuse = x.shape[0]
    probmc = torch.empty((times, numuse, L))
    model.eval()
    bss=get_bs(batch_infer_size, x.shape[0])
    test_loader = DataLoader(CustomDatasetx(x), batch_size=bss, shuffle=False)
    for times_count in range(times):
        probeach = torch.empty((0, L))
        with torch.no_grad():
            set_inference_mode(model, True)
            for xx in test_loader:
                xx = trans_x(xx,device).to(device)
                probthis = F.softmax(model(xx), 1).cpu()
                probeach = torch.cat((probeach, probthis), 0)
            set_inference_mode(model, False)
        probmc[times_count] = probeach
        del probeach
    return probmc.permute(1,2,0)




#### AL ####
def AL_multi_question(get_model_f,device, Xall, Yall, L, trueanswer_fun, locX,
                      AL_max_budget,maxm=0,batch_active_size=0,epochs=10,batch_optimize_size=512,batch_infer_size=512,
                      question_set=[None], question_and_answer=None,initial_model=None,lr=None,
                      AL_max_iteration=10000,print_ALprocess=False, hyper_regular=0,
                      e_and_e=True, explo_rate=0.75,ini_quantile=0.05, threshold_len=6,
                      trans_x=nn.Identity(),layers_to_regularize=None,
                      namepro="",need_MCdrop=None,MCdropnum=40,
                      xtest=None,ytest=None):
    #get min cost of questions and maximum m
    if maxm==0 and question_set[0] is None:
        maxm=1
    min_cost = 10000.
    for question_num in range(len(question_set)):
        if (question_set[question_num][5][1] < min_cost):
            min_cost = question_set[question_num][5][1]
        if maxm!=0 and question_set[0] is not None:
            if maxm<int(question_set[question_num][5][0]):
                maxm=int(question_set[question_num][5][0])

    #check if MCdrop is needed
    if need_MCdrop is None:
        if len(question_num)>0 and question_set[1][5][1]<AL_max_budget:
            need_MCdrop=True
        else:
            need_MCdrop=False

    # Initialize model and parameters.
    # If the initial model parameter is not given, fit the model.
    if question_and_answer is None:
        question_and_answer=torch.empty((0,maxm+3),dtype=torch.int64)
    temp1dvec = question_and_answer[:, 3:].reshape(-1)
    locq=temp1dvec[temp1dvec != -1]
    del temp1dvec
    modelnow=get_model_f(num_classes=L).to(device)

    if initial_model is None:
        build_model_A(model=modelnow,y=Yall[locX], x=Xall[torch.cat((locX,locq))], maxm=maxm,
                      question_set=question_set,question_and_answer=question_and_answer,
                      hyper_regular=hyper_regular, epochs=epochs, batch_size=batch_optimize_size,
                      layers_to_regularize=layers_to_regularize,lr=lr)
    else:
        modelnow.load_state_dict(initial_model)
    set_inference_mode(modelnow, False)

    # record the performance of the initial model
    record_perf_this_iter,zXall = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall, L=L, batch_size=batch_infer_size,
                    needprob=False, needlogits=True, trans_x=trans_x,showprocess=False)
    predproball = F.softmax(zXall,1)
    record_perf=record_perf_this_iter.clone().reshape(1,5)
    record_budget = torch.tensor([0],dtype=torch.float32)
    record_question = torch.tensor([0],dtype=torch.int64)
    record_answer = torch.tensor([0], dtype=torch.int64)
    record_criper = np.ones((1,len(question_set)))*-1
    budget_use = torch.tensor(0.)
    modeling_iteration=True
    last_optimize_budget=torch.tensor(0.)

    # record the performance of the initial model on test data if provided
    if xtest is not None:
        record_perf_test_iter=get_pred_result(model=modelnow, device=device, x=xtest, y=ytest, L=L, batch_size=batch_infer_size,
                        needprob=False, needlogits=False, trans_x=trans_x, showprocess=False)
        record_perf_test=record_perf_test_iter.clone().reshape(1,5)

    # predict the probabilities of all data in the data pool
    if need_MCdrop:
        probmc = get_probmc(model=modelnow, x=Xall, L=L, times=MCdropnum,
                            batch_infer_size=batch_infer_size, trans_x=trans_x,device=device)
    else:
        probmc = predproball

    #Begin active learning
    for ALiteration in range(AL_max_iteration):
        #Check whether the budget is enough to query the cheapest question
        budget_not_enough = (min_cost+budget_use) > (AL_max_budget+1e-7)
        if budget_not_enough:
            break
        t1=time.time()
        #obtain all unlabeled data
        used_set = set((torch.cat((locX, locq))).tolist())
        all_set = set(torch.arange(zXall.shape[0], dtype=torch.int64).tolist())
        candilist = list(all_set - used_set)
        locconsiderall = torch.tensor(candilist, dtype=torch.int64)
        #Screen the data using the exploration and exploitation framework if needed
        if e_and_e:
            locconsider0, threshold_dis = exploration_and_exploitation(x=zXall,
                                   lochave=torch.cat((locX, locq)),device=device,
                                   explo_rate=explo_rate,ini_quantile=ini_quantile,
                                   threshold_len=threshold_len,return_radius=True)
        else:
            locconsider0 = locconsiderall
            threshold_dis = 0
        t2 = time.time()
        #print("time of ee: ", t2 - t1, flush=True)

        # Get the optimal realization for each question
        best_eachq = [None] * len(question_set)
        for question_num in range(len(question_set)):
            t1 = time.time()
            cost_this = question_set[question_num][5][1]
            if (budget_use + cost_this) <= AL_max_budget:
                budget_not_enough = False
                #t1=time.time()
                if question_num==0:
                    qktloc = question_set[question_num][4](probtest=probmc[locconsider0])
                else:
                    qktloc = question_set[question_num][4](probtest=probmc[locconsiderall])
                best_eachq[question_num] = qktloc
            t2=time.time()
            #print("Question ",question_num," use time ",t2-t1,flush=True)

        # Select optimal question
        temp_re = np.ones((1, len(question_set)))*(-1)
        best_perform = -100
        best_num = 0
        t1=time.time()
        for question_num in range(len(question_set)):
            if (best_eachq[question_num] is not None):
                perf = max(best_eachq[question_num][0],0) / (question_set[question_num][5][1])
                perf = perf + 1e-5
                temp_re[0,question_num] = perf
        temp_re+=1e-5
        temp_re[temp_re<0]=0
        temp_re[np.isnan(temp_re)]=0
        best_num = np.random.choice(len(question_set), p=((temp_re**2).squeeze() / (temp_re**2).sum()))
        best_perform = temp_re[0,best_num]
        t2 = time.time()
        record_criper = np.concatenate((record_criper,temp_re))

        #Get the true answer of the queried question
        if best_num==0:
            loctrue = locconsider0[best_eachq[best_num][1]]
        else:
            loctrue = locconsiderall[best_eachq[best_num][1]]
        if len(best_eachq[best_num]) > 2:
            yask = best_eachq[best_num][2]
        else:
            yask = 0
        ans = trueanswer_fun[best_num](loctrue.numpy(), Yall, y=yask)

        # update training data and budget used
        budget_use = budget_use + question_set[best_num][5][1]
        len_m = len(loctrue)
        loctrue = loctrue.reshape(len(loctrue))
        if best_num == 0:
            locX = torch.cat((locX,loctrue))
        else:
            newitem1=torch.tensor([best_num,ans,yask],dtype=torch.int64)
            newitem2=torch.cat([loctrue, torch.full((maxm - len_m,), -1,dtype=torch.int64)])
            newitem=torch.cat((newitem1,newitem2)).reshape(1,3+maxm)
            question_and_answer=torch.cat((question_and_answer,newitem))
            locq=torch.cat((locq,loctrue))

        # Check whether to optimze the model
        # Because under the batch active learning setting, the model will not be updated for every iteration
        if batch_active_size>0:
            modeling_iteration = ((budget_use + min_cost) > (last_optimize_budget+batch_active_size+1e-5)) | ((min_cost + budget_use)>(AL_max_budget+1e-5))
        else:
            modeling_iteration = True

        #If the model needs to be updated, update the model based on the current training set
        if modeling_iteration:
            # Update model
            modeling_iteration = False
            last_optimize_budget += batch_active_size
            t1=time.time()
            if budget_use < (min_cost+batch_active_size+1e-5) and batch_active_size!=0:
                break_num_this=5
            else:
                break_num_this = 5
            build_model_A(model=modelnow, device=device, y=Yall[locX], x=Xall[torch.cat((locX,locq))], maxm=maxm,
                          initial_model=None,question_set=question_set, question_and_answer=question_and_answer,
                          hyper_regular=hyper_regular,epochs=epochs, trans_x=trans_x,lr=lr,
                          return_loss=False, not_optimize=False, batch_size=batch_optimize_size,
                          layers_to_regularize=layers_to_regularize, break_num=break_num_this, return_full_loss=False)
            t2=time.time()
            #print("Training time: ",t2-t1,flush=True)

            # Update predicted probabilities of all data in the data pool
            record_perf_this_iter, zXall = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall, L=L,
                                                           batch_size=batch_infer_size,
                                                           needprob=False, needlogits=True, trans_x=trans_x,
                                                           showprocess=False)
            predproball = F.softmax(zXall, 1)
            if need_MCdrop and (budget_use+min_cost)<=(AL_max_budget+1e-5):
                t1=time.time()
                probmc = get_probmc(model=modelnow, x=Xall, L=L, times=MCdropnum,
                                    batch_infer_size=batch_infer_size, trans_x=trans_x,device=device)
                t2 = time.time()
                #print("MCdrop time: ", t2 - t1, flush=True)
            else:
                probmc = predproball

            # get the performance of the updated model on test data if provided
            if xtest is not None:
                record_perf_test_iter = get_pred_result(model=modelnow, device=device, x=xtest, y=ytest, L=L,
                                                        batch_size=batch_infer_size,
                                                        needprob=False, needlogits=False, trans_x=trans_x,
                                                        showprocess=False)

            if print_ALprocess:
                print(namepro, "budget use ", np.round(budget_use, 3), "/", AL_max_budget,
                      ". ACC: ", np.round(record_perf_this_iter[0].numpy(), 4),
                      ". Loss: ", np.round(record_perf_this_iter[1].numpy(), 4),
                      flush=True)
        # record the performance of the updated model on the data pool and test data if provided
        record_perf = torch.cat((record_perf,record_perf_this_iter.reshape(1,5)),0)
        record_budget = torch.cat((record_budget,budget_use.unsqueeze(0)))
        record_question = torch.cat((record_question,torch.tensor([best_num],dtype=torch.int64)))
        record_answer = torch.cat((record_answer,torch.tensor([ans],dtype=torch.int64)))
        if xtest is not None:
            record_perf_test = torch.cat((record_perf_test,record_perf_test_iter.reshape(1,5)),0)

    # record the performance of the updated model on the data pool and test data if provided
    record_all=torch.cat((record_perf,
                          record_budget.reshape(len(record_budget),1),
                          record_question.reshape(len(record_question),1),
                          record_answer.reshape(len(record_answer),1)),1)
    if xtest is not None:
        record_all = torch.cat((record_all,record_perf_test),1)
    record_criper=torch.tensor(record_criper,dtype=torch.float32)
    record_all = torch.cat((record_all,record_criper),1)
    return record_all


def AL_other_cri(get_model_f, device, Xall, Yall, L, locX,
                 AL_max_budget, criuse, batch_active_size=0, epochs=10, batch_optimize_size=512,batch_infer_size=512,
                 initial_model=None,lr=None, print_ALprocess=False, hyper_regular=0,
                 e_and_e=True, explo_rate=0.75, ini_quantile=0.05, threshold_len=6,
                 trans_x=nn.Identity(), layers_to_regularize=None, namepro="",
                 xtest=None, ytest=None, need_embed=False, temp_use_emb=False):
    modelnow = get_model_f(num_classes=L).to(device)
    if initial_model is None:
        build_model_A(model=modelnow, y=Yall[locX], x=Xall[locX], maxm=1,
                      hyper_regular=hyper_regular, epochs=epochs, batch_size=batch_optimize_size,
                      layers_to_regularize=layers_to_regularize,lr=lr)
    else:
        modelnow.load_state_dict(initial_model)
    set_inference_mode(modelnow, False)

    if need_embed:
        record_perf_this_iter, zXall, emball = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall, L=L,
                                                               batch_size=batch_infer_size,
                                                               needprob=False, needlogits=True, trans_x=trans_x,
                                                               showprocess=False, need_embed=need_embed)
    else:
        record_perf_this_iter, zXall = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall, L=L,
                                                       batch_size=batch_infer_size,
                                                       needprob=False, needlogits=True, trans_x=trans_x,
                                                       showprocess=False)
    predproball = F.softmax(zXall, 1)
    record_perf = record_perf_this_iter.clone().reshape(1, 5)
    record_budget = torch.tensor([0], dtype=torch.float32)
    record_answer = torch.tensor([0], dtype=torch.int64)

    if xtest is not None:
        record_perf_test_iter = get_pred_result(model=modelnow, device=device, x=xtest, y=ytest, L=L,
                                                batch_size=batch_infer_size,
                                                needprob=False, needlogits=False, trans_x=trans_x, showprocess=False)
        record_perf_test = record_perf_test_iter.clone().reshape(1, 5)
    if batch_active_size == 0:
        batch_active_size = 1
    AL_max_iteration = int(AL_max_budget / batch_active_size)

    for ALiteration in range(AL_max_iteration):
        if e_and_e:
            if temp_use_emb:
                locconsider, threshold_dis = exploration_and_exploitation(x=emball,
                                                                          lochave=locX,
                                                                          device=device,
                                                                          explo_rate=explo_rate,
                                                                          ini_quantile=ini_quantile,
                                                                          threshold_len=threshold_len,
                                                                          return_radius=True)
            else:
                locconsider, threshold_dis = exploration_and_exploitation(x=zXall,
                                                                          lochave=locX,
                                                                          device=device,
                                                                          explo_rate=explo_rate,
                                                                          ini_quantile=ini_quantile,
                                                                          threshold_len=threshold_len,
                                                                          return_radius=True)
        else:
            used_set = set((locX).tolist())
            all_set = set(torch.arange(zXall.shape[0], dtype=torch.int64).tolist())
            candilist = list(all_set - used_set)
            locconsider = torch.tensor(candilist, dtype=torch.int64)
            threshold_dis = 0

        if need_embed:
            loctrue = criuse(probtest=predproball[locconsider], sampling_batch=batch_active_size,
                             device=device,embtest=emball[locconsider])
        else:
            loctrue = criuse(probtest=predproball[locconsider], sampling_batch=batch_active_size,
                             device=device)
        loctrue = locconsider[loctrue]

        ans = Yall[loctrue]
        locX = torch.cat((locX, loctrue))
        if ALiteration==0:
            break_num_this=5
        else:
            break_num_this = 5
        build_model_A(model=modelnow, device=device, y=Yall[locX], x=Xall[locX], maxm=1,
                      initial_model=None, hyper_regular=hyper_regular, epochs=epochs, trans_x=trans_x,
                      return_loss=False, not_optimize=False, batch_size=batch_optimize_size,lr=lr,
                      layers_to_regularize=layers_to_regularize, break_num=break_num_this, return_full_loss=False)
        if need_embed:
            record_perf_this_iter, zXall, emball = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall,
                                                                   L=L, batch_size=batch_infer_size,
                                                                   needprob=False, needlogits=True, trans_x=trans_x,
                                                                   showprocess=False, need_embed=need_embed)
        else:
            record_perf_this_iter, zXall = get_pred_result(model=modelnow, device=device, x=Xall, y=Yall, L=L,
                                                           batch_size=batch_infer_size,
                                                           needprob=False, needlogits=True, trans_x=trans_x,
                                                           showprocess=False)
        predproball = F.softmax(zXall, 1)
        record_perf = torch.cat((record_perf, record_perf_this_iter.reshape(1, 5).expand(int(batch_active_size), -1)), 0)
        if xtest is not None:
            record_perf_test_iter = get_pred_result(model=modelnow, device=device, x=xtest, y=ytest, L=L,
                                                    batch_size=batch_infer_size,
                                                    needprob=False, needlogits=False, trans_x=trans_x,
                                                    showprocess=False)

        if print_ALprocess:
            print(namepro, "budget use ", (ALiteration + 1), "/", AL_max_iteration,
                  ". ACC: ", np.round(record_perf_this_iter[0].numpy(), 4),
                  ". Loss: ", np.round(record_perf_this_iter[1].numpy(), 4),
                  flush=True)
        record_budget = torch.cat(
            (record_budget, torch.arange(0, batch_active_size) + 1 + ALiteration * batch_active_size))
        record_answer = torch.cat(
            (record_answer, torch.tensor(ans, dtype=torch.int64).reshape([int(batch_active_size)])))
        if xtest is not None:
            record_perf_test = torch.cat((record_perf_test, record_perf_test_iter.reshape(1, 5).expand(int(batch_active_size), -1)), 0)

    # return model

    record_all = torch.cat((record_perf,
                            record_budget.reshape(len(record_budget), 1),
                            torch.zeros(len(record_budget), 1),
                            record_answer.reshape(len(record_answer), 1)), 1)
    if xtest is not None:
        record_all = torch.cat((record_all, record_perf_test), 1)
    return record_all


