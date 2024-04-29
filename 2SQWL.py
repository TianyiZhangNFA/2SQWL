#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:00:36 2024

@author: Tianyi Zhang
"""

import math
import cmath
from scipy.integrate import quad
import numpy as np

def prob_qw(w_fun,W_w_int,we_real,we_imag,alpha,Beta,theta,t,sigma=1,app_prob=0.999,quad_limit=50):
    '''
    
    Parameters
    ----------
    w_fun : fucntion
        W function on theta and k for the inital state.
    W_w_int : float
        integral results of (w_fun)^2 on (-pi,pi).
    we_real : function
        Real part of w_fun
    we_imag : function
        imaginary part of w_fun.
    alpha : float
        alpha in inital state.
    Beta : float
        beta in inital state.
    theta : float
        theta in Hadamard operator.
    t : int
        time of quantum walk.
    sigma : float, optional
        standard deviation, only valid for truncated Gaussian law. The default is 1.
    app_prob : float, optional
        float number close to 1, representing the proportion of the probability magnitude preserved in this quantum walk algorithm. The default is 0.999.
    quad_limit : int, optional
        number of limit in scipy.integrate.quad function for numerical integral. The default is 50.

    Returns
    -------
    x_i_list : list
        domain or results set of quantum walk at time t.
    psi_final : numpy array
        psi vectors at time t at location in x_i_list.
    prob_results : list
        probability vector of quantum walker appeared at x_i_list at time t.

    '''
    
    
    U=np.array([[np.cos(theta),np.sin(theta)],[np.sin(theta),-1*np.cos(theta)]])
    
    #W_w_int=quad(w2_fun,-1*math.pi, math.pi, args=(theta, sigma),limit=quad_limit)[0]
    
    x=0
    we_real_int_x=quad(we_real,-1*math.pi, math.pi, args=(x,theta, sigma),limit=quad_limit)[0]
    we_imag_int_x=quad(we_imag,-1*math.pi, math.pi, args=(x,theta, sigma),limit=quad_limit)[0]
    acc_mag=(we_real_int_x**2+we_imag_int_x**2)/(2*math.pi*W_w_int)
    while acc_mag<app_prob:
        x+=1
        we_real_int_x_p=quad(we_real,-1*math.pi, math.pi, args=(x,theta, sigma),limit=quad_limit)[0]
        we_imag_int_x_p=quad(we_imag,-1*math.pi, math.pi, args=(x,theta, sigma),limit=quad_limit)[0]
        we_real_int_x_m=quad(we_real,-1*math.pi, math.pi, args=(-1*x,theta, sigma),limit=quad_limit)[0]
        we_imag_int_x_m=quad(we_imag,-1*math.pi, math.pi, args=(-1*x,theta, sigma),limit=quad_limit)[0]
        acc_mag+=(we_real_int_x_p**2+we_imag_int_x_p**2+we_real_int_x_m**2+we_imag_int_x_m**2)/(2*math.pi*W_w_int)
    x_i_list=range(-x,x+1)
    
    psi_initial_i=np.array([[0+0*1j for y in range(2)] for x in range(2*(t+1)+1)])
    psi_initial_i[t+1,:]=np.array([alpha,Beta])
    psi_i=np.copy(psi_initial_i)
    psi_i_c=np.copy(psi_initial_i)
    for j in range(t):
        for ind in range(t+1-j-1,t+1+j+2):
            psi_i[ind,:]=np.add(np.linalg.multi_dot([np.array([[1,0],[0,0]]),U,np.array([list(psi_i_c[ind+1,:])]).T]),np.linalg.multi_dot([np.array([[0,0],[0,1]]),U,np.array([list(psi_i_c[ind-1,:])]).T])).T
        psi_i_c=np.copy(psi_i)
        
    psi_final=np.array([[0+0*1j for y in range(2)] for x in range(2*(t+1)+1)])
    for x_i in x_i_list:
        if x_i>0:
            psi_i_x_i=np.concatenate((np.array([[0,0]]*x_i),psi_i_c[0:(2*(t+1)+1-x_i),]),axis=0)
        elif x_i==0:
            psi_i_x_i=np.copy(psi_i)
        else:
            psi_i_x_i=np.concatenate((psi_i_c[(-1*x_i):(2*(t+1)+1),],np.array([[0,0]]*(-1*x_i))),axis=0)
        psi_final=np.add(psi_final,((quad(we_real,-1*math.pi, math.pi, args=(x_i,theta, sigma),limit=quad_limit)[0]+quad(we_imag,-1*math.pi, math.pi, args=(x_i,theta, sigma),limit=quad_limit)[0]*1j)/np.sqrt(2*math.pi*W_w_int))*psi_i_x_i)
    
    prob_results=[0]*(2*t+1)
    for x_j in range(1,2*(t+1)):
        prob_results[x_j-1]=(psi_final[x_j,]@np.matrix.conjugate(psi_final[x_j,])).real
        
    return x_i_list,psi_final,prob_results
