import numpy as np
from astropy.stats import sigma_clipped_stats
from bayes_opt import BayesianOptimization
from astropy.stats import sigma_clip
from scipy.stats import ks_2samp

def scale_M(init_points,
              n_iter,
              M_crop,
              P_crop,
              boffset,
              bsoften,
              a = 2.5*np.log10(2.718281)
             ):
    
    def KS_loss(soft,
            offset,
            boffset=boffset,
            bsoften=bsoften,
            M_crop=M_crop,
            P_crop=P_crop,
            a = 2.5*np.log10(2.718281)
            ):
    
        M_scaled = a * np.arcsinh((M_crop - offset * boffset) / (soft * bsoften * np.sqrt(a)))
        MKL = np.histogram(M_scaled, density=True, bins=100)[0] + 1e-12
        PKL = np.histogram(P_crop, density=True, bins=100)[0] + 1e-12

        return -ks_2samp(MKL, PKL).statistic
    
    def my_loss(soft,
            offset,
            boffset=boffset,
            bsoften=bsoften,
            M_crop=M_crop,
            P_crop=P_crop,
            a = 2.5*np.log10(2.718281)
            ):
        
        M_scaled = a * np.arcsinh((M_crop - offset * boffset) / (soft * bsoften * np.sqrt(a)))
        
        mse = np.sqrt(np.sum((M_scaled - P_crop)**2))
#         numerator = np.sum((M_scaled - np.mean(M_scaled))*(P_crop - np.mean(P_crop)))
#         denominator = np.sqrt(np.sum((M_scaled - np.mean(M_scaled))**2)*np.sum((P_crop - np.mean(P_crop))**2))
        
        return -mse
    
    pbounds = {'soft': (1e-6, 2),
               'offset': (1e-6, 2),
              }
    
    optimizer = BayesianOptimization(
        
        f=my_loss, 
        pbounds=pbounds,
        random_state=1,
        verbose=False
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    offset = optimizer.max['params']['offset']
    soft = optimizer.max['params']['soft']
    
    return offset, soft
