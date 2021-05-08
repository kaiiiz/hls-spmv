#!/usr/bin/env python
# coding: utf-8

# In[102]:


from scipy.sparse import random
import numpy as np

S = random(256, 256, density=0.3, format="csr", random_state=2906, data_rvs=lambda s: np.random.randint(1, 100, size=s), dtype=np.int32)
np.savetxt('matrix.dat', S.A, delimiter=' ', fmt='%i')


# In[103]:


rows, cols = S.nonzero()
data = np.array([(S[i,j]) for i, j in zip(rows, cols)])


# In[104]:


rows_compressed = [0]
cur_row = 0
cur_row_elmts = 0

for i, j in zip(rows, cols):
    if i > cur_row:
        for _ in range(i - cur_row):
            rows_compressed.append(cur_row_elmts)
        cur_row = i
    cur_row_elmts += 1
rows_compressed.append(cur_row_elmts)


# In[105]:


np.savetxt('rows.dat', np.expand_dims(rows_compressed, 0), delimiter=' ', fmt='%i')
np.savetxt('cols.dat', np.expand_dims(cols, 0), delimiter=' ', fmt='%i')
np.savetxt('data.dat', np.expand_dims(data, 0), delimiter=' ', fmt='%i')


# In[106]:


rows_compressed


# In[107]:


256*256

