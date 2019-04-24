
    max_leaf: Appropriate values are data-dependent and usually varied from 1000 to 10000.
    test_interval: For efficiency, it must be either multiple or divisor of 100 (default value of the optimization interval).
    algorithm: You can select "RGF", "RGF Opt" or "RGF Sib".
    loss: You can select "LS", "Log", "Expo" or "Abs".
    reg_depth: Must be no smaller than 1. Meant for being used with algorithm = "RGF Opt" or "RGF Sib".
    l2: Either 1, 0.1, or 0.01 often produces good results though with exponential loss (loss = "Expo") and logistic loss (loss = "Log"), some data requires smaller values such as 1e-10 or 1e-20.
    sl2: Default value is equal to l2. On some data, l2/100 works well.
    normalize: If turned on, training targets are normalized so that the average becomes zero.
    min_samples_leaf: Smaller values may slow down training. Too large values may degrade model accuracy.
    n_iter: Number of iterations of coordinate descent to optimize weights.
    n_tree_search: Number of trees to be searched for the nodes to split. The most recently grown trees are searched first.
    opt_interval: Weight optimization interval in terms of the number of leaf nodes.
    learning_rate: Step size of Newton updates used in coordinate descent to optimize weights.
