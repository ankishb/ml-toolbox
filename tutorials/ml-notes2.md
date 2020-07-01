
## PCA vs SVD:
1. 
SVD is a numerical method and PCA is an analysis approach (like least squares). You can do PCA using SVD, or you can do PCA doing the eigen-decomposition of 𝑋'𝑋 (or 𝑋𝑋'), or you can do PCA using many other methods, just like you can solve least squares with a dozen different algorithms like Newton's method or gradient descent or SVD etc.

So there is no "advantage" to SVD over PCA because it's like asking whether Newton's method is better than least squares: the two aren't comparable.

2.
From what I understand. SVD has to do with general matrix factorization in some sense. It can be called as a numerical method to do factorization such the we can decompose 𝐴=𝑈𝐷𝑉𝑇

such that D is diagonal and U and V are orthogonal. Thats it.

Now if we formulate the problem of PCA i.e. find all directions of maximum variation and those directions should be orthogonal and try finding the solution. One solution comes out to be the directions being eigen vectors of the covariance matrix. Thats why SVD can be used to find the solution of PCA. The eigen vectors are the principal components and its importance is based on the eigen value of each eigen vector.