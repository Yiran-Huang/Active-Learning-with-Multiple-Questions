import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
from functools import partial

#For BALD, prob is of shape (n,L,K), where n is sample size
#L is class number, and K is MC number.

#### Exchanging ####
def Exchanging_algorithm_includecluster(probtest, deltaen2, fast_screening, m, threshold=0.1, maxiteration=None,
                                        change_fast=None,need_embed=False,embedtest=None):
    m = int(round(m))
    if maxiteration is None:
        maxiteration = 1 * int(m**0.5)
    if m == 1:
        maxiteration = m
    L = probtest.shape[1]
    n = probtest.shape[0]

    # choose best combination for each y
    besten2 = -1000

    for y in range(L):
        breakflag = False
        # find all alternative x
        if probtest.dim()>2:
            alterloc = fast_screening(y=y, probtest=torch.mean(probtest,2), threshold=threshold)
        else:
            alterloc = fast_screening(y=y, probtest=probtest, threshold=threshold)
        # some situation
        # if alternative set is less than m

        if len(alterloc) < m:
            alterloc = np.random.choice(range(probtest.shape[0]), m, replace=False)
            # if alternative set is m
        if len(alterloc) == m:
            locnow = np.copy(alterloc)
            qktnow = [0, y]
            if need_embed:
                en2now = deltaen2(qkt=qktnow, prob=probtest[alterloc, :],embed=embedtest[alterloc, :])
            else:
                en2now = deltaen2(qkt=qktnow, prob=probtest[alterloc, :])
            breakflag = True

            # exchanging
        reorderflag = True
        if not breakflag:
            locnow = np.random.choice(alterloc, m, replace=False)
            qktnow = [0, y]
            if need_embed:
                en2now = deltaen2(qkt=qktnow, prob=probtest[locnow, :],embed=embedtest[locnow, :])
            else:
                en2now = deltaen2(qkt=qktnow, prob=probtest[locnow, :])

        for exchange_count in range(maxiteration):
            if breakflag:
                break
            if reorderflag:
                loc_optimization_order = np.random.choice(range(m), m, replace=False)
                optimiloc = -1
                reorderflag = False
            optimiloc = optimiloc + 1
            if change_fast is None:
                for j in np.setdiff1d(alterloc, locnow):
                    locnew = np.copy(locnow)
                    locnew[optimiloc] = j
                    qktnew = copy.deepcopy(qktnow)
                    if need_embed:
                        en2new = deltaen2(qkt=qktnew, prob=probtest[locnew, :], embed=embedtest[locnew, :])
                    else:
                        en2new = deltaen2(qkt=qktnew, prob=probtest[locnew, :])
                    if en2new > en2now:
                        en2now = np.copy(en2new)
                        locnow = np.copy(locnew)
                        qktnow = copy.deepcopy(qktnew)
                        reorderflag = True
            else:
                loc_candi_exchange=np.setdiff1d(alterloc, locnow)
                loc_change = locnow[optimiloc]
                loc_unchange = locnow[np.setdiff1d(range(m), optimiloc)]
                if need_embed:
                    loc_change_new, en2new = change_fast(y, prob=probtest[loc_unchange],prob_candi=probtest[loc_candi_exchange],
                                                         embed=embedtest[loc_unchange],embed_candi=embedtest[loc_candi_exchange])
                else:
                    loc_change_new, en2new=change_fast(y, prob=probtest[loc_unchange],prob_candi=probtest[loc_candi_exchange])
                if en2new > en2now:
                    en2now = np.copy(en2new)
                    locnow[optimiloc] = loc_candi_exchange[loc_change_new]
                    reorderflag = True

            if (not reorderflag) and optimiloc == (m - 1):
                break
        if en2now > besten2:
            qkt = copy.deepcopy(qktnow)
            besten2 = np.copy(en2now)
            locbest = np.copy(locnow)

    return [besten2, locbest, qkt[1]]


def Exchanging_algorithm_notincludecluster(probtest, deltaen2, fast_screening, m, threshold=0.1, maxiteration=None,
                                           change_fast=None,need_embed=False,embedtest=None):
    m = int(round(m))
    if maxiteration is None:
        maxiteration = 1 * int(m**0.5)
    if m == 1:
        maxiteration = m
    L = probtest.shape[1]
    n = probtest.shape[0]

    # choose best combination for each y
    besten2 = -1000
    breakflag = False
    # find all alternative x
    if probtest.dim() > 2:
        alterloc = fast_screening(probtest=torch.mean(probtest, 2), threshold=threshold)
    else:
        alterloc = fast_screening(probtest=probtest, threshold=threshold)
    # some situation
    # if alternative set is less than m
    if len(alterloc) < m:
        alterloc = np.random.choice(range(probtest.shape[0]), m, replace=False)

    # if alternative set is m
    if len(alterloc) == m:
        locnow = np.copy(alterloc)
        qktnow = [0]
        if need_embed:
            en2now = deltaen2(qkt=qktnow, prob=probtest[alterloc, :], embed=embedtest[alterloc, :])
        else:
            en2now = deltaen2(qkt=qktnow, prob=probtest[alterloc, :])
        breakflag = True

    # exchanging
    reorderflag = True
    if not breakflag:
        locnow = np.random.choice(alterloc, m, replace=False)
        qktnow = [0]
        if need_embed:
            en2now = deltaen2(qkt=qktnow, prob=probtest[locnow, :], embed=embedtest[locnow, :])
        else:
            en2now = deltaen2(qkt=qktnow, prob=probtest[locnow, :])

    for exchange_count in range(maxiteration):
        if breakflag:
            break
        if reorderflag:
            loc_optimization_order = np.random.choice(range(m), m, replace=False)
            optimiloc = -1
            reorderflag = False
        optimiloc = optimiloc + 1
        if change_fast is None:
            for j in np.setdiff1d(alterloc, locnow):
                locnew = np.copy(locnow)
                locnew[optimiloc] = j
                qktnew = copy.deepcopy(qktnow)
                if need_embed:
                    en2new = deltaen2(qkt=qktnew, prob=probtest[locnew, :], embed=embedtest[locnew, :])
                else:
                    en2now = deltaen2(qkt=qktnew, prob=probtest[locnew, :])
                if en2new > en2now:
                    en2now = np.copy(en2new)
                    locnow = np.copy(locnew)
                    qktnow = copy.deepcopy(qktnew)
                    reorderflag = True
        else:
            loc_candi_exchange = np.setdiff1d(alterloc, locnow)
            loc_change = locnow[optimiloc]
            loc_unchange = locnow[np.setdiff1d(range(m), optimiloc)]
            if need_embed:
                loc_change_new, en2new = change_fast(y=0, prob=probtest[loc_unchange],prob_candi=probtest[loc_candi_exchange],
                                                     embed=embedtest[loc_unchange],embed_candi=embedtest[loc_candi_exchange])
            else:
                loc_change_new, en2new = change_fast(y=0, prob=probtest[loc_unchange],prob_candi=probtest[loc_candi_exchange])
            if en2new > en2now:
                en2now = np.copy(en2new)
                locnow[optimiloc] = loc_candi_exchange[loc_change_new]
                reorderflag = True

        if (not reorderflag) and optimiloc == (m - 1):
            break

    qkt = copy.deepcopy(qktnow)
    besten2 = np.copy(en2now)
    locbest = np.copy(locnow)
    return [besten2, locbest]


def exchange_random(probtest, deltaen2, fast_screening, m, threshold=0.1,
                    maxiteration=None,change_fast=None,need_embed=False,embedtest=None):
    locbest = np.random.choice(range(probtest.shape[0]), 1)
    besten2 = 10
    return [besten2, locbest]


#### Q1 ####
def entropy_loss_q1(a, qkt, prob=None, model=None):
    y = qkt[1]
    if prob is None:
        prob = model(qkt[0])
    pra1 = prob[:, y].prod()
    pra0 = 1 - pra1
    return -a * torch.log(pra1 + 1e-5) - (1 - a) * torch.log(pra0 + 1e-5)

def KL_q1(probtrue, probpred, qkt):
    y = qkt[1]
    pra1true = probtrue[:, y].prod() + 1e-5
    pra0true = 1 - pra1true + 1e-5
    pra1pred = probpred[:, y].prod() + 1e-5
    pra0pred = 1 - pra1pred + 1e-5
    return pra1true * torch.log(pra1true / pra1pred + 1e-5) + pra0true * torch.log(
        pra0true / pra0pred + 1e-5)

def BALD_q1(qkt, prob):
    y = qkt[1]
    prob_yes_mc = torch.prod(prob[:, y],0)
    prob_no_mc = 1 - prob_yes_mc
    prob_yes_E = prob_yes_mc.mean()
    prob_no_E = 1 - prob_yes_E

    H_E = - prob_yes_E * torch.log(prob_yes_E + 1e-5) - prob_no_E * torch.log(prob_no_E + 1e-5)
    H_mc = (- prob_yes_mc * torch.log(prob_yes_mc + 1e-5) - prob_no_mc * torch.log(prob_no_mc + 1e-5)).mean()

    out = H_E - H_mc
    return out.numpy()

def BALD_q1_change(y, prob,prob_candi):
    prob_yes_mc = torch.prod(prob[:, y],0) * prob_candi[:,y]
    prob_no_mc = 1-prob_yes_mc
    prob_yes_E = torch.mean(prob_yes_mc,1)
    prob_no_E = 1 - prob_yes_E

    H_E = - prob_yes_E * torch.log(prob_yes_E + 1e-5) - prob_no_E * torch.log(prob_no_E + 1e-5)
    H_mc = (- prob_yes_mc * torch.log(prob_yes_mc + 1e-5) - prob_no_mc * torch.log(prob_no_mc + 1e-5))
    H_mc = torch.mean(H_mc,1)

    out = H_E - H_mc
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()

def fast_screen_q1(y, probtest, threshold=0.1):
    for tt in np.append(np.linspace(threshold, 0, 4), -1):
        aa = torch.tensor(np.where(probtest[:, y] > tt)).squeeze()
        if (aa.dim() > 0):
            if (aa.shape[0] > 0):
                break
    return aa.squeeze()

#### Q2 ####
def entropy_loss_q2(a, qkt, prob=None, model=None):
    y = qkt[1]
    if prob is None:
        prob = model(qkt[0])
    pra0 = (1 - prob[:, y]).prod()
    pra1 = 1 - pra0
    return -a * torch.log(pra1 + 1e-5) - (1 - a) * torch.log(pra0 + 1e-5)


def KL_q2(probtrue, probpred, qkt):
    y = qkt[1]
    pra0true = (1 - probtrue[:, y]).prod() + 1e-5
    pra1true = 1 - pra0true + 1e-5
    pra0pred = (1 - probpred[:, y]).prod() + 1e-5
    pra1pred = 1 - pra0pred + 1e-5
    return pra1true * torch.log(pra1true / pra1pred + 1e-5) + pra0true * torch.log(
        pra0true / pra0pred + 1e-5)

def BALD_q2(qkt, prob):
    y = qkt[1]
    prob_no_mc = torch.prod(1-prob[:, y],0)
    prob_yes_mc = 1 - prob_no_mc
    prob_no_E = prob_no_mc.mean()
    prob_yes_E = 1 - prob_no_E

    H_E = - prob_yes_E * torch.log(prob_yes_E + 1e-5) - prob_no_E * torch.log(prob_no_E + 1e-5)
    H_mc = (- prob_yes_mc * torch.log(prob_yes_mc + 1e-5) - prob_no_mc * torch.log(prob_no_mc + 1e-5)).mean()

    out = H_E - H_mc
    return out.numpy()

def BALD_q2_change(y, prob,prob_candi):
    prob_no_mc = torch.prod(1-prob[:, y],0) * (1 - prob_candi[:,y])
    prob_yes_mc = 1-prob_no_mc
    prob_no_E = torch.mean(prob_no_mc,1)
    prob_yes_E = 1 - prob_no_E

    H_E = - prob_yes_E * torch.log(prob_yes_E + 1e-5) - prob_no_E * torch.log(prob_no_E + 1e-5)
    H_mc = (- prob_yes_mc * torch.log(prob_yes_mc + 1e-5) - prob_no_mc * torch.log(prob_no_mc + 1e-5))
    H_mc = torch.mean(H_mc,1)

    out = H_E - H_mc
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()

def fast_screen_q2(y, probtest, threshold=0.1):
    for tt in np.append(np.linspace(threshold, 0, 4), -1):
        #aa = torch.tensor(np.where(((1 - probtest[:, y]) > tt) & (probtest[:, y] > tt))).squeeze()
        aa = torch.tensor(np.where((1 - probtest[:, y]) > tt)).squeeze()
        if (aa.dim() > 0):
            if (aa.shape[0] > 0):
                break
    return aa.squeeze()


#### Q4 ####
def entropy_loss_q4(a, qkt, prob=None, model=None):
    if prob is None:
        prob = model(qkt[0])
    en = -torch.log(prob[:, a] + 1e-5)
    return torch.sum(en)

def KL_q4(probtrue, probpred, qkt):
    return torch.sum(probtrue * torch.log(probtrue / (probpred + 1e-5) + 1e-5))

def BALD_q4(qkt, prob):
    prob=prob.squeeze(0)
    prob_E = torch.mean(prob,1)

    H_E = (- prob_E * torch.log(prob_E + 1e-5)).sum()
    H_mc = - prob * torch.log(prob + 1e-5)
    H_mc = torch.mean(torch.sum(H_mc,0))

    out = H_E - H_mc
    return out.numpy()

def BALD_q4_change(y, prob,prob_candi):
    prob_E = torch.mean(prob_candi , 2)

    H_E = - prob_E * torch.log(prob_E + 1e-5)
    H_E = torch.sum(H_E,1)
    H_mc = - prob_candi * torch.log(prob_candi + 1e-5)
    H_mc = torch.mean(torch.sum(H_mc,1),1)

    out = H_E - H_mc
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()

def entropy_q4(qkt, prob):
    out = (- prob * torch.log(prob + 1e-5)).sum()
    return out.numpy()

def entropy_q4_change(y, prob,prob_candi):
    out = torch.sum(-prob_candi * torch.log(prob_candi + 1e-5),1)

    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()

def leastconfident_q4(qkt, prob):
    out = 1-torch.max(prob)
    return out.numpy()

def leastconfident_q4_change(y, prob, prob_candi):
    out = 1-torch.max(prob_candi,1)[0]
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()

def minmargin_q4(qkt, prob):
    max2 = torch.topk(prob, 2)[0]
    out = 1 - (max2[:,0] - max2[:,1])
    return out.numpy()

def minmargin_q4_change(y, prob, prob_candi):
    max2 = torch.topk(prob_candi, 2)[0]
    out = 1 - (max2[:, 0] - max2[:, 1])
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()

def fast_screen_q4(probtest, threshold=0.1):
    for tt in np.append(np.linspace(threshold, 0, 4), -1):
        a1, _ = torch.max(probtest, dim=1)
        aa = torch.tensor(np.where((1 - a1) > tt)).squeeze()
        if(aa.dim()>0):
            if (aa.shape[0] > 0):
                break
    return aa.squeeze()


#### Create question ####
def Create_question(loss_function, change_fast_function, deltaen2function,
                    screen_function, exchang_function, parameter,need_embed=False):
    m = parameter[0]
    cost = parameter[1]
    threshold = parameter[2]
    find_optimal_qkt = partial(exchang_function,deltaen2=deltaen2function, fast_screening=screen_function,
                                  m=m, threshold=threshold,change_fast=change_fast_function,
                                  need_embed=need_embed)
    out = [loss_function, change_fast_function,
           deltaen2function, screen_function, find_optimal_qkt, parameter]

    return out

def Q1_create_BALD(m, cost, threshold):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q1, change_fast_function=BALD_q1_change,
                        deltaen2function=BALD_q1, screen_function=fast_screen_q1,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q2_create_BALD(m, cost, threshold):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q2, change_fast_function=BALD_q2_change,
                        deltaen2function=BALD_q2, screen_function=fast_screen_q2,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q4_create_BALD(m, cost, threshold):
    m = 1
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=BALD_q1_change,
                        deltaen2function=BALD_q4, screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )

def Q4_create_entropy(m, cost, threshold):
    m = 1
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=entropy_q4_change,
                        deltaen2function=entropy_q4, screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )

def Q4_create_leastconfident(m, cost, threshold):
    m = 1
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=leastconfident_q4_change,
                        deltaen2function=leastconfident_q4, screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )

def Q4_create_minmargin(m, cost, threshold):
    m = 1
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=minmargin_q4_change,
                        deltaen2function=minmargin_q4, screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )

raAL = Create_question(loss_function=entropy_loss_q4, change_fast_function=None,
                       deltaen2function=entropy_q4, screen_function=None, exchang_function=exchange_random,
                       parameter=np.array([1, 1, 0]))
def GainLinear_q4(qkt, prob,power=1):
    out = 2 * ( prob * (1-prob)**power).sum()
    return out.numpy()

def GainLinear_q4_change(y, prob,prob_candi,power=1):
    out = 2 * torch.sum( prob_candi * ( 1 - prob_candi )**power,1)
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()
def Q4_create_GainLinear(m, cost, threshold,power=1):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=partial(GainLinear_q4_change,power=power),
                        deltaen2function=partial(GainLinear_q4,power=power), screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )

#### Real answer ####
def Q1_answer(loc, Ytest, y):
    alllabel = Ytest[loc]
    if (torch.sum(alllabel == y) == len(alllabel)):
        return 1
    else:
        return 0

def Q2_answer(loc, Ytest, y):
    alllabel = Ytest[loc]
    if (torch.sum(alllabel == y) > 0):
        return 1
    else:
        return 0

def Q4_answer(loc, Ytest, y=0):
    return Ytest[loc]


#### other function ####
def torch_quantile(x, q, dim=None, interpolation='linear'):
    x_np = x.detach().cpu().numpy()
    if dim is None:
        result = np.quantile(x_np, q, interpolation=interpolation)
    else:
        result = np.quantile(x_np, q, axis=dim, interpolation=interpolation)
    return torch.tensor(result, dtype=x.dtype, device=x.device)
def quantile_index(x, q, dim=None):
    if dim is None:
        x = x.flatten()
        sorted_indices = torch.argsort(x)
        rank = int(round(q * (x.size(0) - 1)))
        return sorted_indices[rank]
    else:
        sorted_indices = torch.argsort(x, dim=dim)
        n = x.size(dim)
        rank = int(round(q * (n - 1)))
        idx = torch.index_select(sorted_indices, dim, torch.tensor([rank], device=x.device))
        return idx.squeeze(dim)


def GainTVq_q1(qkt, prob,q=0.3):
    y = qkt[1]
    pyesall = torch.prod(prob[:,y],0)
    loc = quantile_index(pyesall, q, dim=None)
    prob_yes = pyesall[loc]
    prob_no = 1 - prob_yes
    p1=prob[:, :,loc]

    Gain_yes = torch.sum(1-p1[:,y])
    Gain_no = p1[:,y].min()

    out = prob_yes*Gain_yes + prob_no*Gain_no
    return out.numpy()

def GainTVq_q1_change(y, prob,prob_candi,q=0.3):
    pyesall = torch.prod(prob[:, y],0) * prob_candi[:,y]
    loc = quantile_index(pyesall, q, dim=1)
    prob_yes = pyesall[torch.arange(prob_candi.shape[0]),loc]
    prob_no = 1 - prob_yes
    p1 = prob[:,:,loc]
    p2 = prob_candi[torch.arange(prob_candi.shape[0]),:,loc]

    sumyes = torch.sum(1-p1[:,y],0)
    Gain_yes = (1-p2[:,y]) + sumyes
    Gain_no = p2[:,y]
    if len(prob)>0:
        minno = p1[:,y].min(dim=0)[0]
        temp=minno-Gain_no
        Gain_no = (temp<0)*temp+Gain_no

    out = prob_yes*Gain_yes + prob_no*Gain_no
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()


def GainTVq_q2(qkt, prob, q=0.3):
    y = qkt[1]
    pnoall = torch.prod(1 - prob[:, y], 0)
    loc = quantile_index(pnoall, q, dim=None)
    prob_no = pnoall[loc]
    prob_yes = 1 - prob_no
    p1= prob[:, :,loc]

    Gain_no = (p1[:,y]).sum()
    Gain_yes = (1-p1[:,y]).min()

    out = prob_yes * Gain_yes + prob_no * Gain_no
    return out.numpy()


def GainTVq_q2_change(y, prob, prob_candi, q=0.3):
    pnoall = torch.prod(1 - prob[:, y], 0) * (1 - prob_candi[:, y])
    loc = quantile_index(pnoall, q, dim=1)
    prob_no = pnoall[torch.arange(prob_candi.shape[0]), loc]
    prob_yes = 1 - prob_no
    p1 = prob[:,:,loc]
    p2 = prob_candi[torch.arange(prob_candi.shape[0]),:,loc]

    sumno = torch.sum(p1[:,y],0)
    Gain_no = (p2[:,y]) + sumno
    Gain_yes = 1-p2[:,y]
    if len(prob)>0:
        minyes = (1-p1[:,y]).min(dim=0)[0]
        temp=minyes-Gain_yes
        Gain_yes = (temp<0)*temp+Gain_yes

    out = prob_yes * Gain_yes + prob_no * Gain_no
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()


def GainTVq_q4(qkt, prob,power=1):
    p1 = torch.mean(prob, dim=2)
    out = torch.sum(p1 * (1-p1)**power , 1).squeeze()
    return out.numpy()


def GainTVq_q4_change(y, prob, prob_candi,power=1):
    p1 = torch.mean(prob_candi, dim=2)
    out = torch.sum(p1 * (1-p1)**power , 1)
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()


def Q1_create_GainTVq(m, cost, threshold,q=0.3):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q1, change_fast_function=partial(GainTVq_q1_change,q=q),
                        deltaen2function=partial(GainTVq_q1,q=q), screen_function=fast_screen_q1,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q2_create_GainTVq(m, cost, threshold,q=0.3):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q2, change_fast_function=partial(GainTVq_q2_change,q=q),
                        deltaen2function=partial(GainTVq_q2,q=q), screen_function=fast_screen_q2,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q4_create_GainTVq(m, cost, threshold,power=1):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=partial(GainTVq_q4_change,power=power),
                        deltaen2function=partial(GainTVq_q4,power=power), screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )


def GainTVnoq_q1(qkt, prob):
    y = qkt[1]
    prob_yes = torch.prod(prob[:,y])
    prob_no = 1 - prob_yes

    Gain_yes = torch.sum(1-prob[:,y])
    Gain_no = prob[:,y].min()

    out = prob_yes*Gain_yes + prob_no*Gain_no
    return out.numpy()

def GainTVnoq_q1_change(y, prob,prob_candi):
    prob_yes = torch.prod(prob[:, y]) * prob_candi[:,y]
    prob_no = 1 - prob_yes

    sumyes = torch.sum(1-prob[:,y],0)
    Gain_yes = (1-prob_candi[:,y]) + sumyes
    Gain_no = prob_candi[:,y]
    if len(prob)>0:
        minno = prob[:,y].min()
        temp=minno-Gain_no
        Gain_no = (temp<0)*temp+Gain_no

    out = prob_yes*Gain_yes + prob_no*Gain_no
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy() , out[loc_best_candi].numpy()


def GainTVnoq_q2(qkt, prob):
    y = qkt[1]
    prob_no = torch.prod(1 - prob[:, y])
    prob_yes = 1 - prob_no

    Gain_no = (prob[:,y]).sum()
    Gain_yes = (1-prob[:,y]).min()

    out = prob_yes * Gain_yes + prob_no * Gain_no
    return out.numpy()


def GainTVnoq_q2_change(y, prob, prob_candi):
    prob_no = torch.prod(1 - prob[:, y]) * (1 - prob_candi[:, y])
    prob_yes = 1 - prob_no

    sumno = torch.sum(prob[:,y])
    Gain_no = (prob_candi[:,y]) + sumno
    Gain_yes = 1-prob_candi[:,y]
    if len(prob)>0:
        minyes = (1-prob[:,y]).min()
        temp=minyes-Gain_yes
        Gain_yes = (temp<0)*temp+Gain_yes

    out = prob_yes * Gain_yes + prob_no * Gain_no
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()


def GainTVnoq_q4(qkt, prob,power=1):
    out = torch.sum(prob * (1-prob)**power , 1).squeeze()
    return out.numpy()


def GainTVnoq_q4_change(y, prob, prob_candi,power=1):
    out = torch.sum(prob_candi * (1-prob_candi)**power , 1)
    loc_best_candi = torch.argmax(out)
    return loc_best_candi.numpy(), out[loc_best_candi].numpy()


def Q1_create_GainTVnoq(m, cost, threshold,q=0.3):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q1, change_fast_function=GainTVnoq_q1_change,
                        deltaen2function=GainTVnoq_q1, screen_function=fast_screen_q1,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q2_create_GainTVnoq(m, cost, threshold,q=0.3):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q2, change_fast_function=GainTVnoq_q2_change,
                        deltaen2function=GainTVnoq_q2, screen_function=fast_screen_q2,
                        exchang_function=Exchanging_algorithm_includecluster, parameter=parameter)
    )

def Q4_create_GainTVnoq(m, cost, threshold,power=1):
    parameter = np.array([m, cost, threshold])
    return (
        Create_question(loss_function=entropy_loss_q4, change_fast_function=partial(GainTVnoq_q4_change,power=power),
                        deltaen2function=partial(GainTVnoq_q4,power=power), screen_function=fast_screen_q4,
                        exchang_function=Exchanging_algorithm_notincludecluster, parameter=parameter)
    )
