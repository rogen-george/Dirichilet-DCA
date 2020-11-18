#!/usr/bin/env python
# coding: utf-8

from scipy.stats import dirichlet
from scipy.special import digamma
from scipy.special import polygamma
import numpy as np
import math
from numpy import errstate,isneginf,array

from statistics import median
import random
from sklearn.preprocessing import normalize

alpha = np.array([2, 2, 2, 2])

# Estimating the mean and precision from a dirichilet distribution 
# https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf


# Generate synthetic data from Dirichilet 
data = dirichlet.rvs(alpha, size = 10000)  

# Estimate the precision given the mean of the data

def estimate_precision( data ):
    # Fix the mean 
    mean = [ 1 / len(data[0]) ] * len(data[0]) 

    dimensions = len(mean)
    N = len( data )
    p_k = np.zeros(dimensions)

    # Number of iterations for Newton Update
    iterations = 100

    # Calculate the value of log p_k
    for dimension in range( dimensions ):
        with errstate(divide='ignore'):
            p_k[dimension] = np.sum ( np.log( data[:, dimension] ) ) / N
        p_k[isneginf(p_k)]=0

    denominator = 0
    for i, m_k in enumerate(mean):
        denominator += m_k * ( p_k[i] - np.log(m_k) ) 

    # Calculate the initial value of precision s
    numerator = (dimensions - 1) / 2 
    initial_s = - 1* numerator / denominator

    s = initial_s
    for i in range ( iterations ):
        #print (s)
        second_term = 0
        third_term = 0
        d2_term = 0
        for i, m_k in enumerate(mean):
            second_term += m_k * digamma( s * m_k )
            third_term += m_k * p_k[i]
            d2_term += ( (m_k) ** 2 ) * polygamma(1, s * m_k)

        d_log_likelihood =   N * ( digamma(s) - second_term + third_term )
        d2_log_likelihood =  N * ( polygamma(1, s) - d2_term )

        # Update the value of s after each iteration 
        s = 1 / ( (1 / s) + (1 / (s ** 2) ) *  ( ( 1 / d2_log_likelihood ) * d_log_likelihood ) ) 

    return (s)

def balanced_rearrangement_matrices(n, k):
    n_n = n
    k_k = k

    N_K = n_n/k_k

    x = np.zeros((k , n))
    matrices = []

    # Generate matrices whose each row sum to 1 and each colum sum to 1/k 
    # where k is the reduced dimensions
    
    while ( len(matrices) < 50 ):
        n = n_n
        k = k_k
        x = np.zeros((k_k , n_n))
        n_by_k = n_n / k_k

        i = 0

        for i in range(n - 1):
            # First row - First col
            if ( i < k_k - 1):    
                n_by_k = N_K
                x[i,i:] = np.random.random(n)
                n_by_k = ( n_by_k - np.sum( x[i, 0:i] ) )
                x[i,i:]  =  ( x[i,i:]  / np.sum( x[i,i:] ) ) * n_by_k

                # First column
                x[(i + 1):,i] = np.random.random(k - 1)
                t = ( 1 - np.sum( x[0:(i + 1),i] ) )
                x[(i + 1):,i] = ( x[(i + 1):,i] / np.sum( x[(i + 1):,i] ) ) * t 

                n -= 1
                k -= 1

            else:
                break

        for j in range( i , n_n ):
            x[i][j] = 1 - np.sum( x[:,j] )

        if ( np.amin(x) > 0):
            matrices.append(x)


    return (matrices) 


# Takes data and a target dimension k to reduce the feature space to
def genetic_algorithm( data , k = 3):
    
    n = len( data[0] )
    iterations = 50
    
    matrix_population = balanced_rearrangement_matrices(n, k)
    
    for _ in range(iterations):  
        dirichilet_correlation = []
        
        for matrix in matrix_population:
            reduced_data = np.matmul( matrix , data.transpose() ).transpose()

            new_matrix = normalize( reduced_data, norm = 'l1', axis = 0 )
            dirichilet_correlation.append( estimate_precision( reduced_data ) )
        
            updated_population = []
        min_correlation = dirichilet_correlation.index( min(dirichilet_correlation) )
        # Add the matrix with the min population
        
        #if not np.all( matrix_population[min_correlation] == 0 ):
            #updated_population.append( matrix_population[min_correlation] )
        median_dc = median(dirichilet_correlation)
        print ( "median", median_dc)
        if median_dc == 0 :
            return
        dc_updated = np.array ( list( ( map( lambda x : min( x / median_dc, 1 ), dirichilet_correlation) ) ) )
        with errstate(divide='ignore'):
            fitness = - 1 * np.log( dc_updated )
        fitness[isneginf(fitness)]= 0
        fitness = np.nan_to_num(fitness)

        # Add to new population if fitness > 0
        for matrix in range( len(matrix_population)):
            if fitness[matrix] > 0 and not np.all( matrix == 0 ):
                updated_population.append(matrix_population[matrix]) 
          
        for _ in range( len(matrix_population) // 2 ):
            parent1 = random.choice(range(len(matrix_population)))
            parent2 = random.choice(range(len(matrix_population)))
            child = ( matrix_population[parent1] * fitness[parent1] ) + ( matrix_population[parent2] * fitness[parent2] )
            if not np.all( child == 0 ):
                updated_population.append( child )
        #print ("Updated Population ", updated_population)
        
        matrix_population = updated_population
        print (fitness)
        
        if not matrix_population:
            return

genetic_algorithm(data)

