import matplotlib.pyplot as plt
import numpy as np





def main():
    
    print ("Im here")
  
    ##################################################################
    #######   example of three curves in one graph        ############
    ##################################################################
    #===========================================================================
    # x = np.arange(5)
    # y = np.exp(x)
    # fig1, ax1 = plt.subplots()
    # ax1.plot(x, y)
    # ax1.set_title("Axis 1 title")
    # ax1.set_xlabel("X-label for axis 1")
    # ax1.set_ylabel("y-label for axis 1")
    # w = np.cos(x)
    # ax1.plot(x, w) # can continue plotting on the first axis
    # z = np.sin(x)
    # ax1.plot(x,z)
    #===========================================================================
    
    
    #######     My graphs   ##################
    fig1, ax1 = plt.subplots()
    x= [10,14, 18, 22, 26, 30, 34, 36, 40, 44, 48, 52]
    y= [0.044, 0.068, 0.079, 0.118, 0.253, 0.341, 0.362, 0.410 ,0.422,0.434, 0.443, 0.442] #random init
    w= [0.021, 0.054, 0.077, 0.102, 0.156, 0.213, 0.271, 0.280 ,0.311,0.322, 0.321, 0.322] #word2vec init
    z= [0.031, 0.038, 0.052, 0.067, 0.078, 0.123, 0.142, 0.185 ,0.238,0.249, 0.252, 0.251] #sym-patterns init
    
    ax1.set_title("")
    ax1.set_xlabel("Training Time (hours)")
    ax1.set_ylabel("Spearman Correlation")
    ax1.plot(x, y, lw=2, label="uniform init.") #-0.1, 0.1
    ax1.plot(x, w, lw=2, label="word2vec emb.")
    ax1.plot(x, z, lw=2, label="sym-pattern emb.")

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #===========================================================================
    #####  Fig2 epochs       #####################
    ####################################################
    fig2, ax2 = plt.subplots()
    x= [1,2,3,4,5,6,7,8,9,10]
    y= [0.011, 0.017, 0.021, 0.072, 0.09, 0.171, 0.21, 0.21 ,0.22, 0.210] #random init
    
    ax2.set_title("")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Spearman Correlation")
    ax2.plot(x, y, lw=4, label="uniform init.") #-0.1, 0.1
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    #===========================================================================
    #####  Fig3 embeddings       #####################
    ####################################################
    fig3, ax3 = plt.subplots()
    x= [50, 100, 200, 300, 400, 500, 600, 700, 800]
    y= [0.280, 0.356, 0.421, 0.439, 0.442, 0.4421, 0.4423, 0.4421 ,0.4421] #random init
    
    ax3.set_title("")
    ax3.set_xlabel("Embedding Size")
    ax3.set_ylabel("Spearman Correlation")
    ax3.plot(x, y, lw=4, label="uniform init.") #-0.1, 0.1
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
    
    
    #===========================================================================
    plt.show()

if __name__ == '__main__':
    main()
