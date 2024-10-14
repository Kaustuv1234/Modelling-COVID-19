import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np


# this function makes sure that the parameters stay within given constraints 
def projection(params):
    N = 70000000
    
    # keep beta positive
    if params[0] < 0:
        params[0] = 0
        
    # keep R(0) between 15.6% and 36%
    if 0.156*N >= params[4]:
        params[4] = 0.156*N
    elif 0.36*N <= params[4]:
        params[4] = 0.36*N
    
    # keep CIR(0)) between 12.0 and 30.0
    if 12 >= params[5]:
        params[5] = 12
    elif params[5] >= 30:
        params[5] = 30
        
    return params

# returns the data for fitting the model
def get_training_data():

    #read data from file
    filepath = '../COVID19_data.csv'
    data = pd.read_csv(filepath)
    
    # store the day number, where t = 0 starts from 16 March 2021
    data['t'] = (pd.to_datetime(data['Date']) - datetime.datetime(2021, 3, 16)).dt.days

    # keep necessary columns
    req = data[['Date', 'Confirmed', 'Tested', 'First Dose Administered', 't']]

    # T is the average number of tests done during the past 7 days (i.e., during t − 7 to t − 1)
    req.insert(2, 'T', (req.Tested.rolling(8).sum() - req.Tested) / 7)
    
    # running seven day average of ∆confirmed(t)
    req.insert(2, 'del_confirmed', req['Confirmed'].diff())
    req.insert(2, 'c_bar_t', (req.del_confirmed.rolling(8).sum() - req.del_confirmed)/7)
    
    # finding ∆V(t)
    req.insert(2, 'del_V', req['First Dose Administered'].diff())

    # keep the data from 16 March 2021 to 26 April 2021
    df = req[req['t'] >= 0]
    temp = df[df['t'] < 42]
    return temp[['t', 'del_V', 'c_bar_t', 'T']]


# calculate running seven day average 
def running_average(a):
    ae_bar = []
    
    # take average of the days available, for t = 0,1,2,3,4,5,6
    for i in range(7):
        x = np.mean(a[:i+1])
        if x < 0:
            x = 1
        ae_bar.append(x)
        
    # take the average of last 7 days, for t >= 7 
    for i in range(7, 42):
        x = np.mean(a[i-7:i])
        if x < 0:
            x = 1
        ae_bar.append(x)
        
    return np.array(ae_bar)


# It takes [beta, S(0), E(0), I(0), R(0), CIR(0)] as parameters and returns a list containing 
# S(t),E(t),I(t),R(t) for 0 ≤ t ≤ 41 
def SEIRV_model(initial_values):

    # initial values
    beta = initial_values[0]

    # for storing S(t), E(t), I(t), R(t) values 
    S = [initial_values[1]]
    E = [initial_values[2]]
    I = [initial_values[3]]
    R = [initial_values[4]]
    
    # constants
    alpha = 1/5.8
    gamma = 1/5
    epsilon = 0.66
    N = 70000000

    # find S(t),E(t),I(t),R(t)
    for t in range(0, 41):

        # conditions for ∆W(t) in problem description
        dW = 0
        if t <= 30:
            dW = R[0] / 30

        # equations
        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        # when S(t) is negative, scale the remaining values so that the total sum is 70000000
        if S[t] + dS < 0:
            S.append(0)
            scale = N/(E[t] + dE + I[t] + dI+ R[t] + dR)
            E.append((E[t] + dE)*scale)
            I.append((I[t] + dI)*scale)
            R.append((R[t] + dR)*scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)
            
    return [S,E,I,R]


# loss function implementation 
def loss_function(params):
    # constants 
    alpha = 1/5.8

    # CIR(t) = CIR(0) * T(t0) / T(t) 
    CIR = params[5] * T[0] * np.reciprocal(T)
    
    # find E using the SEIRV model
    E = SEIRV_model(params)[1]
    
    # calculate α∆e(t)
    ae = alpha * np.divide(E, CIR)  
    
    # find the running seven day average of α∆e(t)
    ae_bar = running_average(ae) 
    
    s = np.log(c_bar) - np.log(ae_bar)
    sq_err = np.square(s).sum()
    return sq_err / 42


# returns the gradient of the loss function 
def gradient(params):
    [beta, S0, E0, I0, R0, CIR0] = params
    grad = []

    # perturb beta on either side by ±0:01, perturb CIR(0) on either side by ±0:1, and all other parameters by ±1
    grad.append((loss_function([beta + 0.01, S0, E0, I0, R0, CIR0]) - loss_function([beta - 0.01, S0, E0, I0, R0, CIR0])) / 0.02)
    grad.append((loss_function([beta, S0 + 1, E0, I0, R0, CIR0]) - loss_function([beta, S0 - 1, E0, I0, R0, CIR0])) / 2)
    grad.append((loss_function([beta, S0, E0 + 1, I0, R0, CIR0]) - loss_function([beta, S0, E0 - 1, I0, R0, CIR0])) / 2)
    grad.append((loss_function([beta, S0, E0, I0 + 1, R0, CIR0]) - loss_function([beta, S0, E0, I0 - 1, R0, CIR0])) / 2)
    grad.append((loss_function([beta, S0, E0, I0, R0 + 1, CIR0]) - loss_function([beta, S0, E0, I0, R0 - 1, CIR0])) / 2)
    grad.append((loss_function([beta, S0, E0, I0, R0, CIR0 + 0.1]) - loss_function([beta, S0, E0, I0, R0, CIR0 - 0.1])) / 0.2)

    return np.array(grad)

# gradient descent implementation
def gradient_descent(params):
    # initialize parameters
    delta = 0.01
    j = 0

    training_data = get_training_data()
    
    # make c_bar, T, dV global so that other funcions can access them
    # running seven day average of ∆confirmed(t)
    global c_bar 
    c_bar = training_data['c_bar_t'].to_numpy()

    # T is the average number of tests done during the past 7 days
    global T 
    T = training_data['T'].to_numpy()

    # T is the average number of tests done during the past 7 days
    global dV 
    dV = training_data['del_V'].to_numpy()

    loss = loss_function(params)
    while (loss > delta):
        # use projection so that the parameters stay within the constraints
        params = projection(params - gradient(params) / (j+1))
        loss = loss_function(params)
        j += 1
        
    print('GRADIENT DESCENT: iterations:', j, 'loss = ', loss)
    return params


# return ∆confirmed(t) values from 16 March 2021 onwards
def new_reported_cases():
    #read data from file
    filepath = '../COVID19_data.csv'
    data = pd.read_csv(filepath)
    
    # store the day number, where t = 0 starts from 16 March 2021
    data['t'] = (pd.to_datetime(data['Date']) - datetime.datetime(2021, 3, 16)).dt.days

    # keep necessary columns
    req = data[['Confirmed', 't']]

    # find the difference in confirmed cases between successive days
    req.insert(2, 'del_confirmed', req['Confirmed'].diff())

    # keep the data from 16 March 2021 onwards
    df = req[req['t'] >= 0]
    
    return df['del_confirmed'].to_list()


# shows the number of predicted and reported new cases till 31 Dec 2021
def plot_cases(result1, result2, result3, result4, result5, CIR0):
    # get average CIR from the training period
    CIR_list = CIR0 * T[0] * np.reciprocal(T) 
    CIR = sum(CIR_list) / 42
    
    # predictions for E till 31 December 2021
    E1 = result1[1]
    E2 = result2[1]
    E3 = result3[1]
    E4 = result4[1]
    E5 = result5[1]
    
    alpha = 1 / 5.8
    predicted_case1 = alpha * np.array(E1) / CIR
    predicted_case2 = alpha * np.array(E2) / CIR
    predicted_case3 = alpha * np.array(E3) / CIR
    predicted_case4 = alpha * np.array(E4) / CIR
    predicted_case5 = alpha * np.array(E5) / CIR
    t = range(291)
    
    # from 21 Sept 2021 to 31 Dec 2021, make the number of reported cases = 0
    reported_cases = new_reported_cases()
    for i in range(188, 291):
        reported_cases.append(0)
        
    plt.figure(figsize=(12, 8))
    
    # plot predicted and reported new cases
    plt.plot(t, predicted_case1)
    plt.plot(t, predicted_case2)
    plt.plot(t, predicted_case3)
    plt.plot(t, predicted_case4)
    plt.plot(t, predicted_case5)
    plt.plot(t, reported_cases)
    
    # plot settings
    legend_name = ['open loop β', 'open loop 2β/3', 'open loop β/2', 'open loop β/3', 'closed loop control', 'reported cases']
    plt.legend(legend_name, loc ="upper right", fontsize = 13)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('cases', fontsize=15)
    plt.show()
    
    
# shows the evolution of the fraction of the susceptible population 
def plot_S_fraction(result1, result2, result3, result4, result5, CIR0):
    # predictions for E till 31 December 2021
    S1 = result1[0]
    S2 = result2[0]
    S3 = result3[0]
    S4 = result4[0]
    S5 = result5[0]
    
    N = 70000000
    t = range(291)
    plt.figure(figsize=(12, 8))
    plt.xlim([0, 400])
    
    # plot fraction of the susceptible population 
    plt.plot(t, np.array(S1)/N)
    plt.plot(t, np.array(S2)/N)
    plt.plot(t, np.array(S3)/N)
    plt.plot(t, np.array(S4)/N)
    plt.plot(t, np.array(S5)/N)
    
    legend_name = ['open loop β', 'open loop 2β/3', 'open loop β/2', 'open loop β/3', 'closed loop control']
    plt.legend(legend_name, loc ="upper right", fontsize = 13)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('fraction of susceptible population', fontsize = 15)
    plt.show()


# predict new cases for different values of beta
def open_loop_control(params):
    # constants
    alpha = 1/5.8
    gamma = 1/5
    epsilon = 0.66
    N = 70000000
    [beta, S0,E0,I0,R0,CIR0] = params

    # for storing S(t), E(t), I(t), R(t) values    
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    # for t < 42 use the given data for ∆V(t), and for remaining use 200000 per day
    dV = get_training_data()['del_V'].to_list()
    for t in range(42, 290):
        dV.append(200000)
    
    # make predictions till t = 290 i.e.(31 Dec 2021)
    for t in range(290): 
        
        # conditions for ∆W(t) in problem description
        dW = 0
        if t <= 30:
            dW = R[0] / 30
        if t >= 180:
            dW = (R[t - 179]-R[t - 180]) + epsilon * dV[t]

        # equations
        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        # when S(t) or R(t) is negative, scale the remaining values so that the total sum is 70000000
        if S[t] + dS < 0:
            S.append(0)
            scale = N/(E[t] + dE + I[t] + dI+ R[t] + dR)
            E.append((E[t] + dE)*scale)
            I.append((I[t] + dI)*scale)
            R.append((R[t] + dR)*scale)
        elif R[t] + dR < 0:
            R.append(0)
            scale = N/(E[t] + dE + I[t] + dI+ S[t] + dS)
            E.append((E[t] + dE)*scale)
            I.append((I[t] + dI)*scale)
            S.append((S[t] + dS)*scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)
        
    return [S,E,I,R]


# initial_values contain [beta, S(0), E(0), I(0), R(0), CIR(0)]
def closed_loop_control(params):
    [beta1, S0,E0,I0,R0,CIR0] = params
    
    # number of vaccinations per day from 16 March 2021 to 26 April 2021
    dV = get_training_data()['del_V'].to_list()
    for t in range(42, 290):
        dV.append(200000)
    
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    # constants
    alpha = 1/5.8
    gamma = 1/5
    epsilon = 0.66
    N = 70000000
    
    # t = 290 is 31 Dec 2021
    for t in range(290): 
        beta = beta1
        if t % 7 == 0 and t >= 42:
            if sum(I[t-7:t])/7 < 10000:
                beta = beta1
            elif sum(I[t-7:t])/7 < 25000:
                beta = 2*beta1/3
            elif sum(I[t-7:t])/7 < 100000:
                beta = beta1/2
            else:
                beta = beta1/3
        
        # conditions for ∆W(t) in problem description
        dW = 0
        if t <= 30:
            dW = R[0] / 30
        if t >= 180:
            dW = (R[t - 179]-R[t - 180]) + epsilon * dV[t-180]

        # equations
        dS = -beta * S[t] * I[t] / N - epsilon * dV[t] + dW
        dE = beta * S[t] * I[t] / N - alpha * E[t]
        dI = alpha * E[t] - gamma * I[t]
        dR = gamma * I[t] + epsilon * dV[t] - dW

        # when S(t) is negative, scale the remaining values so that the total sum is 70000000
        if S[t] + dS < 0:
            S.append(0)
            scale = N/(E[t] + dE + I[t] + dI+ R[t] + dR)
            E.append((E[t] + dE)*scale)
            I.append((I[t] + dI)*scale)
            R.append((R[t] + dR)*scale)
        # when R(t) is negative, scale the remaining values so that the total sum is 70000000
        elif R[t] + dR < 0:
            R.append(0)
            scale = N/(E[t] + dE + I[t] + dI+ S[t] + dS)
            E.append((E[t] + dE)*scale)
            I.append((I[t] + dI)*scale)
            S.append((S[t] + dS)*scale)
        else:
            S.append(S[t] + dS)
            E.append(E[t] + dE)
            I.append(I[t] + dI)
            R.append(R[t] + dR)
            
    return [S,E,I,R]



if __name__ == '__main__':
    
    '''MODELLING'''
    # initial values
    N = 70000000
    beta = 3.41243564864
    e = 0.0016
    i = 0.0039
    r = 0.32
    s = 1 - r - e - i
    CIR0 = 29.054
    
    initial_values = np.array([beta, s*N, e*N, i*N, r*N, CIR0])
    best_params = gradient_descent(initial_values)
    print('best_params', list(best_params))
    
    
    '''OPEN LOOP CONTROL'''
    params = [0.49639073007972884, 47228999.99999971, 97999.9999962104, 272999.99998911645, 22399999.999999844, 29.50884124522309]
    [beta1, S0,E0,I0,R0,CIR0] = params
    
    beta = beta1
    result1 = open_loop_control([beta, S0,E0,I0,R0,CIR0])
    # plot_cases(result, beta, CIR0)
    # plot_S_fraction(result, beta)
    
    beta = 2 * beta1 / 3
    result2 = open_loop_control([beta, S0,E0,I0,R0,CIR0])
    # plot_cases(result, beta, CIR0)
    # plot_S_fraction(result, beta)

    beta = beta1 / 2
    result3 = open_loop_control([beta, S0,E0,I0,R0,CIR0])
    # plot_cases(result, beta, CIR0)
    # plot_S_fraction(result, beta)

    beta = beta1 / 3
    result4 = open_loop_control([beta, S0,E0,I0,R0,CIR0])
    # plot_cases(result, beta, CIR0)
    # plot_S_fraction(result, beta)
    
    
    '''CLOSED LOOP CONTROL'''
    result5 = closed_loop_control(best_params)
    plot_cases(result1, result2, result3, result4, result5, CIR0)
    plot_S_fraction(result1, result2, result3, result4, result5, CIR0)
    
    
    
    '''
    INITIAL PARAMETERS
    beta = 3.41243564864
    e = 0.0016
    i = 0.0039
    r = 0.32
    s = 0.6745
    CIR0 = 29.054
    
    GRADIENT DESCENT: iterations: 34 loss =  0.007057469495405773
    best_params [0.470504737336695, 47214999.99999966, 111999.99999546476, 272999.99998825626, 22399999.999999817, 29.549793811281727]    
    '''
    
    
    
    