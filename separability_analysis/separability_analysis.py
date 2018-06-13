import pandas as pd
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt


class separability_analysis:

    def __init__(self,dataset_path):

        self._data = pd.read_csv(dataset_path, header=None)

        self._total_bands = self._data.shape[1] - 1

        self._classwise_groups = self._data.groupby([self._total_bands])

        self._mean_vectors = np.array(self._classwise_groups.mean())

        self._cov_vectors = list()

        self._n_classes = len(self._data[self._total_bands].unique())

        for i in range(self._n_classes):
            self._cov_vectors.append(np.array(self._classwise_groups.get_group(i+1).cov()))

        self._corr_vectors = list()
        
        for i in range(self._n_classes):
            self._corr_vectors.append(np.array(self._classwise_groups.get_group(i+1).corr()))

    def _uncorrelated_bands(self,class_one, class_two, bands):
    
        freq = np.zeros((self._total_bands),dtype=np.int)
        
        for i in range(self._total_bands):
            for j in range(i,self._total_bands,1):
                if i!=j:
                    if self._corr_vectors[class_one][i,j] > 0.99 :
                        freq[i]+=1
                        freq[j]+=1
                    if self._corr_vectors[class_two][i,j] > 0.99 :
                        freq[i]+=1
                        freq[j]+=1
                        
        bands_to_remove = [i for i in range(len(freq)) if freq[i] > 1]
        new_bands = list(set(bands) - set(bands_to_remove))
        
        return new_bands

    def _jm(self,class_1, class_2, reduced_bands):
    
        bands = self._uncorrelated_bands(class_1,class_2,reduced_bands)

        mean_1 = self._mean_vectors[class_1][bands]
        mean_2 = self._mean_vectors[class_2][bands]
        cov_1 = self._cov_vectors[class_1][bands][:,bands]
        cov_2 = self._cov_vectors[class_2][bands][:,bands]

        mean_diff = mean_1 - mean_2
        mean_cov = ( cov_1 + cov_2 ) / 2
        
        jm_mean_dist = 0.125 * np.dot( np.dot(mean_diff.T , lin.inv(mean_cov)), mean_diff )

        np.seterr(divide='ignore')

        jm_cov_dist = 0.5*np.log( lin.det(mean_cov) / np.sqrt(lin.det(cov_1)*lin.det(cov_2) ))

        alpha = jm_mean_dist + jm_cov_dist

        jm_dist = np.sqrt(2*(1-np.exp(-1*alpha)))
        
        return jm_dist

    def JM_distance_mat(self, bands):

        size = self._n_classes

        jm_mat = np.zeros(shape=(size,size),dtype=np.float)

        for i in range(size):
            for j in range(size):
                if i!=j:
                    jm_mat[i,j] = self._jm(i,j,bands)
                else:
                    jm_mat[i,j] = 0

        # Sorting inorder of JM Distance
        dtype = [('c1',int),('c2',int),('sep',float)]
        pairwise = []
        avg = 0
        cnt = 0

        for i in range(size):
            for j in range(i+1,size):
                pairwise.append((i+1,j+1,jm_mat[i,j]))
                avg += jm_mat[i,j]
                cnt += 1
        avg = avg / cnt

        print("\nAverage JM Distance : ",avg,'\n\n')

        pairwise = np.array(pairwise, dtype=dtype)
        pairwise_sorted = np.sort(pairwise, order='sep')

        plt.figure(figsize=(size+2,size),dpi=80)
        heatmap = plt.pcolor(jm_mat,cmap='gray')

        for y in range(size):
            for x in range(size):
                plt.text(x + 0.5, y + 0.5, '%.4f' % jm_mat[x, y],
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        plt.yticks(np.arange(size)+0.5,np.arange(size)+1)
        plt.xticks(np.arange(size)+0.5,np.arange(size)+1)
        plt.colorbar(heatmap)
        plt.title('Class to class JM-Distance')

        plt.show()

        return pairwise_sorted
