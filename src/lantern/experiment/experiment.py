import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

from lantern.dataset.dataset import Dataset
from lantern.model.model import Model



@attr.s()
class Experiment:

    dataset: Dataset = attr.ib()
    model: Model = attr.ib()
    
    
    def prediction_table(self, 
                         mutations_list=None, # input list of substitutions to predict. If None, the full input data for the experiment is used.
                         max_variants=None, # Only used if mutations_list is None. The max number of input substitutions to predict
                         uncertainty_samples=50, # Number of random draws used to estimate uncertainty of predictions.
                         batch_size=500, # batch size used in calculating predictions and uncertainties. If zero, then the method runs all the predictions in a single batch.
                         uncertainty=False, # Boolean to indicate whether or not to include uncertainties in the output table. If batch_size is zero, this parameter is ignored.
                         predict_from_z=False, # If true, the method ignores the mutations_list and predicts based on an input set of latent-space (Z) vectors
                         z_input=None, # Required if predict_from_z is True. Array/list of latent-space (Z) vectors. 
                         #                   z_input.shape should be N x D, where N is the number of z-vectors to predict and D is the number of dimensions in the latent space.
                         verbose=False): # Whether or not to print extra output.
    
        dataset = self.dataset
        tok = dataset.tokenizer
        model = self.model
        phenotypes = dataset.phenotypes
        errors = dataset.errors
        
        # Start with list of string representations of the mutations in each variant to predict
        #     default is to use the variants in dataset
        if mutations_list is None:
            sub_col = dataset.substitutions
            # in case the substitutions entry for the WT variants is marked as np.nan (it should be an empty string)
            mutations_list = list(dataset.df[sub_col].replace(np.nan, ""))
            
            if max_variants is not None:
                mutations_list = mutations_list[:max_variants]
            
        else:
            df = pd.DataFrame({'substitutions':mutations_list})
            for c in list(phenotypes) + list(errors):
                df[c] = 0
            dataset = Dataset(df, phenotypes=phenotypes, errors=errors)
        
        if type(mutations_list) is not list:
            mutations_list = list(mutations_list)
        
        if (batch_size == 0) or predict_from_z:     
            # Convert from list of mutation strings to one-hot encoding
            X = tok.tokenize(*mutations_list)
            
            # The tokenize() method returns a 1D tensor if len(mutations_list)==1
            #     but here, we always want a 2D tensor
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            
            if predict_from_z:
                with torch.no_grad():
                    Z = torch.Tensor(z_input)
                z_order = model.basis.order
                z_re_order = np.array([np.where(z_order==n)[0][0] for n in range(len(z_order))])
                Z = Z[:, z_re_order]
            else:
                # Get Z coordinates (latent space) for each variant
                Z = model.basis(X)
            
            # Get predicted mean phenotype and variance as a function of Z coordinates
            # 
            # The next line is equivalent to: f = model(X): f = model(X) is equivalent to f = model.forward(X), which is equivalent to f = model.surface(model.basis(X)).
            f = model.surface(Z) 
            with torch.no_grad():
                Y = f.mean.numpy() # Predicted phenotype values
                Yerr = np.sqrt(f.variance.numpy())
                Z = Z.numpy() # latent space coordinates
        else:
            Y = [] # Predicted phenotype values
            Z = [] # latent space coordinates
            if uncertainty:
                Yerr = []
                Yerr_0 = []
                Zerr = []
                        
            mutations_list_list = [mutations_list[pos:pos + batch_size] for pos in range(0, len(mutations_list), batch_size)]
            
            j = 0
            if verbose: print(f'Number of batches: {len(mutations_list_list)}')
            for mut_list in mutations_list_list:
                if verbose: print(f'Batch: {j}')
                # _x is the one-hot encoding for the batch.
                _x = tok.tokenize(*mut_list)
                if len(_x.shape) == 1:
                    _x = _x.unsqueeze(0)
                
                _z = model.basis(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]
                
                _f = model.surface(_z)
                
                with torch.no_grad():
                    _y = _f.mean.numpy()
                    _yerr_0 = np.sqrt(_f.variance.numpy())
                    _z = _z.numpy()
                    
                    if uncertainty:
                        f_tmp = torch.zeros(uncertainty_samples, *_y.shape)
                        z_tmp = torch.zeros(uncertainty_samples, *_z.shape)
                        model.train()
                        for n in range(uncertainty_samples):
                            z_samp = model.basis(_x)
                            z_tmp[n, :, :] = z_samp
                            f_samp = model(_x)
                            samp = f_samp.sample()
                            if samp.ndim == 1:
                                samp = samp[:, None]

                            f_tmp[n, :, :] = samp
                        
                        _yerr = f_tmp.std(axis=0)
                        _zerr = z_tmp.std(axis=0)

                        model.eval()
                                
                Y += list(_y)
                Z += list(_z)
                if uncertainty:
                    Yerr += list(_yerr)
                    Yerr_0 += list(_yerr_0)
                    Zerr += list(_zerr)
                
                j += 1
            Y = np.array(Y)
            Z = np.array(Z)
            if uncertainty:
                Yerr = np.array(Yerr)
                Yerr_0 = np.array(Yerr_0)
                Zerr = np.array(Zerr)
            
        # Fix ordering of Z dimensions from most to least important
        Z = Z[:, model.basis.order]
        
        if uncertainty and (not predict_from_z) and (batch_size != 0):
            Zerr = Zerr[:, model.basis.order]
            Yerr = np.max([Yerr, Yerr_0], axis=0) # make sure the error estimate returned is at least as big as the GP error at fixed Z
        
        # Make the datafream to return
        if predict_from_z:
            df_return = pd.DataFrame({'z1':Z.transpose()[0]})
        else:
            df_return = pd.DataFrame({'substitutions':mutations_list})
        
        # Add predicted phenotype columns
        df_columns = dataset.phenotypes
        df_columns = [x.replace('-norm', '') for x in df_columns]
        for c, y in zip(df_columns, Y.transpose()):
            df_return[c] = y
        if uncertainty:
            for c, yerr in zip(df_columns, Yerr.transpose()):
                df_return[f'{c}_err'] = yerr
            
        # Add columns for Z coordinates
        for i, z in enumerate(Z.transpose()):
            df_return[f'z_{i+1}'] = z
        if uncertainty and not predict_from_z:
            for i, zerr in enumerate(Zerr.transpose()):
                df_return[f'z_{i+1}_err'] = zerr
        
        return df_return
    
    
    def dim_variance_plot(self, ax=None, include_total=False, figsize=[4, 4], **kwargs):
        # Plots the variance for each of the dimensions in the latent space - used to identify which dimonesions are "importaant"
    
        model = self.model

        mean = 1 / model.basis.qalpha(detach=True).mean[model.basis.order]
        z_dims = [n + 1 for n in range(len(mean))]
        
        if ax is None:
            plt.rcParams["figure.figsize"] = figsize
            fig, ax = plt.subplots()
        
        ax_twin = ax.twiny()
        
        ax_twin.plot(z_dims, mean, "-o")

        ax_twin.set_xlabel("Z dimension")
        ax_twin.set_xticks(z_dims)
        ax_twin.set_ylabel("variance")

        mn = min(mean.min(), 1e-4)
        mx = mean.max()
        z = torch.logspace(np.log10(mn), np.log10(mx), 100)
        ax.plot(
            invgammalogpdf(z, torch.tensor(0.001), torch.tensor(0.001)).exp().numpy(),
            z.numpy(),
            c="k",
            zorder=0,
        )
        ax.set_xlabel("prior probability")

        ax.set_yscale('log')
        
    def latent_space_plot(self,
                          df_plot=None, # DataFrame with phenotypes and z-coordinates for the scatterplot. If None, the mutations_list is used to get a prediction_table().
                          mutations_list=None, # input list of substitutions to predict. If None, the full input data for the experiment is used.
                          z_dims=[1,2],
                          phenotype=None,
                          xlim=None, ylim=None, 
                          fig_ax=None, 
                          figsize=[5, 5],
                          colorbar=True,
                          cbar_kwargs={},
                          scatter_alpha = 0.2,
                          contours=True,
                          contour_grid_points=100,
                          contour_kwargs={},
                          **kwargs):
        
        dataset = self.dataset
        if df_plot is None:
            df = self.prediction_table(mutations_list=mutations_list)
        else:
            df = df_plot
        
        if phenotype is None:
            phenotype = dataset.phenotypes[0].replace('-norm', '')
        
        if fig_ax is None:
            plt.rcParams["figure.figsize"] = figsize
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis'
        if 's' not in kwargs:
            kwargs['s'] = 9
            
        x = df[f'z_{z_dims[0]}']
        y = df[f'z_{z_dims[1]}']
        c = df[phenotype]
        
        im = ax.scatter(x, y, c=c, **kwargs)
        
        ax.set_xlabel(f'$Z_{z_dims[0]}$')
        ax.set_ylabel(f'$Z_{z_dims[1]}$')
        
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, **cbar_kwargs)
            cbar.ax.set_ylabel(phenotype, rotation=270, labelpad=20)
        
        if contours:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            rect = patches.Rectangle((xlim[0], ylim[0]), 
                                     xlim[1] - xlim[0], 
                                     ylim[1] - ylim[0], 
                                     edgecolor='none', 
                                     facecolor='w', 
                                     alpha=scatter_alpha)
            ax.add_patch(rect)
            
            x_points = np.linspace(*xlim, contour_grid_points)
            y_points = np.linspace(*ylim, contour_grid_points)
            x_points, y_points = np.meshgrid(x_points, y_points)
            
            # Fill out the rest of the z-vectors with zeros
            x_flat = x_points.flatten()
            y_flat = y_points.flatten()
            
            z_list = np.zeros((len(x_flat), len(self.model.basis.order)))
            
            z_list.transpose()[z_dims[0]-1] = x_flat
            z_list.transpose()[z_dims[1]-1] = y_flat
            
            df_flat = self.prediction_table(predict_from_z=True, z_input=z_list, uncertainty=False)
            
            p_flat = df_flat[phenotype]
            p_points = np.split(p_flat, contour_grid_points)
            
            if 'cmap' not in contour_kwargs:
                contour_kwargs['cmap'] = 'viridis'
            
            ax.contour(x_points, y_points, p_points, **contour_kwargs)
            
            return fig, ax

def invgammalogpdf(x, alpha, beta):
    return alpha * beta.log() - torch.lgamma(alpha) + (-alpha - 1) * x.log() - beta / x

