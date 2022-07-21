# Shapelet with Neural Network

## **Symbol Table**

| | Itself | Length | Number | Iterator |
| -: | :-: | :-: | :-: | :-: |
| Time series | $s$ | $l$ | $n$ | $i$ |
| Subsequences | $t$ | $m$ | $(l-m+1)$ | $p$ |
| Shapelets | $v$ | $m$ | $k$ | $j$ |
| Categories | $\textbf{y}_i$ | - | $N_\textrm{cat}$ | $c$ |

## SOINN: [Yang, 2016](https://doi.org/10.1109/ICTAI.2016.0071)
- Learning candidates: **SOINN** (online unsupervised) learns the topological structure & reports clusters
- Data transformation: $\textrm{similarity} = e^{-\alpha \frac{\textrm{sdist}(t, v)}{T_P}}$, $T_P$ is a threshold learned by SOINN (denser data, small $T_P$)
- Shapelets selection: learn a *sparse* weight matrix with **SVM** (L1-reg): $\textrm{loss} = C\sum_i \max(0, 1-y_i w^T x_i)^2 + \sum_j |w_j|$
    - Candidates with associated weights being all zeros are removed

## Triple shapelet network: [Ma, 2019](https://doi.org/10.1109/ICDM.2019.00155) -- **Categorial & dynamic, 3 scope levels**
- Current issues: classification accuracy vs. distinguishing power of the minority class; fixed shapelets
- "Triple": dataset-level (softmax outputs), category-level (binary auxiliary classifier), sample-level (shapelet generator)
- **Dataset-level**: $k$ shapelets; use shapelet transform to generate feature $\textbf{h}_i^D$
- **Category-level**: $k' = \lceil\frac{k}{N_\textrm{cat}}\rceil$ shapelets *per category*; the $j^\textrm{th}$ category has feature $\textbf{h}_{i, j}^C$ ($N_\textrm{cat}$ in total)
- **Sample-level**: Convolve over concat. subseqs.: $v_j = \textbf{W}_j \ast \textbf{O} + b_j$ is the $j^\textrm{th}$ generated shapelet ($k$ in total), feature $\textbf{h}_i^S$
    - $L$ is the length, $P$ is the number of subsequences; $\textbf{W}_j \in \Bbb{R}^{5 \times P}$, $\textbf{O} \in \Bbb{R}^{L \times P}$, $b_j \in \Bbb{R}$
- **Classification loss**: pred. value $P = \textrm{softmax}(\textbf{W}^\textrm{out}\textrm{concat}(h_i^D, h_i^C, h_i^S))$; $\textbf{W}^\textrm{out} \in \Bbb{R}^{N_\textrm{cat} \times 3k}$, $h_i^\ast \in \Bbb{R}^k$
- **Auxiliary loss**: pred. value $P = \sigma(\textbf{w} \cdot \textbf{h})$; $\textbf{w} \in \Bbb{R}^{N_\textrm{cat} \times k'}$, $\textbf{h} \in \Bbb{R}^{k'}$ (no concat., otherwise deprecated to dataset-level)
- Learnables: dataset-level, category-level shapelets; $\textbf{W}_j$, $b_j$ (sample-level conv.), $\textbf{W}^\textrm{out}$ (cls. loss), $\textbf{w}$ (aux. loss)

![19ma](img/19ma.png)

## Adversarial regulatization: [Wang, 2019](https://doi.org/10.48550/arXiv.1906.00917) -- **GAN, 3 optimization steps**
- A discriminator is trained to distinguish between shapelets and subsequences
- **Optimize classifier step**: Dot product (softmaxed distance vector, learn weight vector) -> pred value, optimize $\theta_c$
- **Optimize discriminator step**: $\textrm{loss} = ||v_j||^2 - ||t_j||^2 + \lambda(||\partial(\epsilon t_j + (1-\epsilon) v_j)||^2 - 1)$ (***??????***), optimize $\theta_d$
- **Optimize shapelet step**: $\textrm{loss} = -||v_j||^2$, optimize $\theta_s$ *(Note: $\theta_s$ are the shapelets themselves!)*
- Losses: classifier - multi-class cross entropy, discriminator - WGAN-GP, shapelet - normal length

![19wang](img/19wang.png)

## Adversarial dynamic shapelet network: [Ma, 2020](https://doi.org/10.1609/AAAI.v34i04.5948) -- **Global conv. kernels, 3 losses**
- Current issues: shapelets deviate from the subsequences ("adversarial"); fixed shapelets ("dynamic")
- **Shapelet generator**: same as ("Sample-level", TSN)
- **Dynamic shapelet transformation**: $P = \textrm{softmax}(\textbf{W}^\textrm{out}h_i)$; $\textbf{W}^\textrm{out} \in \Bbb{R}^{N_\textrm{cat} \times k}$, $h_i \in \Bbb{R}^k$ - *fully dynamic???*
- **Adversarial training strategy**: "two-player minimax game"; $D$ is a *two-layer conv-nn*
- **Diversity Regularization Term**: $\textrm{similarity} = e^{-||\Delta d||}$
- Losses: $L_\textrm{cls}$ - cross entropy, $L_\textrm{adv}$ - $-\frac{1}{nk}\sum_{i=1}^n\sum_{j=1}^k\log(\textrm{discriminator}(t_i, v_j))$, $L_\textrm{div}$ - Frobenius norm
- ### Complete pseudo-code
    - Hyperparameters: $n_E$ (num. epochs), $n_B$ (batch size), $k$, $m$, $\eta$ (learning rate), $\lambda_\textrm{Adv}$, $\lambda_\textrm{Div}$ (regularization params.)
    - Learnables: $\textbf{W}_j$, $b_j$ (conv. kernels), $\textbf{W}_\textrm{out}$ (linear classification layer), $\theta_D$ (conv-nn), $\theta_\textrm{ADSN}$ (ensemble of params.)
    - For $n_E$ epochs and $\lceil\frac{n}{n_B}\rceil$ batches
        - For time series $i = 1 : n_B$
            - Concatenate subsequences into matrix $\textbf{O}_i \in \Bbb{R}^{m \times (l-m+1)}$
            - For "dynamic" shapelets $j = 1 : k$
                - Compute shapelet $v_{i, j} = \textbf{W}_j \ast \textbf{O}_i + b_j$ ($\textbf{W}_j \in \Bbb{R}^{\textrm{width} \times (l-m+1)}$, $b_j \in \Bbb{R}$)
                - Compute distance $h_{i, j} = \min_{p = 1 : (l-m+1)} |v_{i, j} - t_{i, p}|$
            - Compute predicted category $\hat{\textbf{y}}_i = \textrm{Softmax}(\textbf{W}_\textrm{out} \textbf{h}_i)$ ($\textbf{W}_\textrm{out} \in \Bbb{R}^{N_\textrm{cat} \times k}$, $\textbf{h}_i \in \Bbb{R}^k$, $\hat{\textbf{y}}_i \in \Bbb{R}^{N_\textrm{cat}}$)
            - Compute similarity matrix $\textbf{G}_i [j, j'] = e^{-|v_{i, j} - v_{i, j'}|}$ ($\textbf{G}_i \in \Bbb{R}^{k \times k}$; vectorization may apply)
        - For $\textrm{numIter}$ cycles of learning discriminator $D$
            - Compute $L_D = -\sum_i \sum_p \log D(t_{i, p}) -\sum_i \sum_j \log D(1-v_{i, j})$ ($D(\cdot)$ - similarity to real subseqs.)
            - SGD the parameters: $\theta_D = \theta_D - \eta \frac{\partial L_D}{\partial \theta_D}$
        - Compute $\textrm{Loss}_\textrm{Classification} = -\frac{1}{n_B} \sum_i \sum_c \textbf{y}_{i, c} \hat{\textbf{y}}_{i, c}$ (maximize predicted value at correct category)
        - Compute $\textrm{Loss}_\textrm{Adversarial} = -\frac{1}{n_B k} \sum_i \sum_j \log D(v_{i, j})$ (maximize shapelets' similarity to subseqs.)
        - Compute $\textrm{Loss}_\textrm{Diversity} = ||\textrm{Concat} (\textbf{G}_1, \ldots, \textbf{G}_{n_B})||^2_F$ (minimize shapelets' similarity to each other)
        - Compute $L_\textrm{ADSN} = \textrm{Loss}_\textrm{Cls} + \lambda_\textrm{Adv} \textrm{Loss}_\textrm{Adv} + \lambda_\textrm{Div} \textrm{Loss}_\textrm{Div}$
        - SGD the parameters: $\theta_\textrm{ADSN} = \theta_\textrm{ADSN} - \eta \frac{\partial L_\textrm{ADSN}}{\partial \theta_\textrm{ADSN}}$

![20ma](img/20ma.png)

## Convolutional transfer: [DeSouza, 2021](https://doi.org/10.1007/978-3-030-91699-2_23)
- General idea: *decompose* the time series (O) into shapelets (S) and noise (N), train *two classifiers* separately (pre-initialization)
- O w/rand, S w/rand, O w/S, N w/rand, O w/N, S w/N, N w/S

## Multivar ShapeNet: [Li, 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17018) -- **Mdc-CNN, 4 unsup. "clustering" losses**
- **Multi-length-input dilated causal CNN**: embed shapelet candidates of different lengths into a *unified space* for selection
    - Residual blocks + (global maxpool + linear) --> *shared weights*
    - *Dilated*: blanks in the hidden layers, achieves exponentially large receptive field
    - *Causal*: no future value <==> no connections with "negative slope" (fig. bottom right)
- ***Unsupervised* representation training**: *cluster-wise triplet loss* considers the distance between (anchor, Ps, Ns)
    - "Single positive" is unstable; CWT minimizes $\mathcal{D}_{AP}$, $\mathcal{D}_{PP}$, $\mathcal{D}_{NN}$, maximizes $\mathcal{D}_{AN}$
    - $\mathcal{D}_{AP} = \frac{1}{||\textbf{x}^+||} \sum_i ||f(x)-f(x_i^+)||^2$, $\mathcal{D}_{AN} = \frac{1}{||\textbf{x}^-||} \sum_i ||f(x)-f(x_i^-)||^2$ - $f(\cdot)$ is Mdc-CNN ^
    - $\mathcal{D}_{PP} = \max_{1 \le i, j \le ||\textbf{x}^+||} ||f(x_i^+)-f(x_j^+)||^2$, $\mathcal{D}_{PP} = \max_{1 \le i, j \le ||\textbf{x}^-||} ||f(x_i^-)-f(x_j^-)||^2$
    - $L_\textrm{CWT} = \log \frac{\mathcal{D}_{AP}+\mu}{\mathcal{D}_{AN}} + \lambda (\mathcal{D}_{PP}+\mathcal{D}_{NN})$ - $\mu$ is the "enforced P-N margin"; $\lambda$ is hyperparam.
    - *Note: how can we call this approach "unsupervised" if we can tell the positive samples from the negative ones?*
- **Multivar shapelet transform**: Compute *representative and diversified* final shapelets rather than using all embeddings
    - Final shapelets should be (1) *close* to the centroid of a *large* cluster and (2) *far* from other clusters - select top-$k$
        - $\mathcal{U}(f(x_i)) = \beta \frac{\log \textrm{ClusterSize}(f(x_i))}{\log \max_i \textrm{ClusterSize}(f(x_i))} + (1-\beta) \frac{\log \sum_j ||f(x_i)-f(x_j)||^2}{\log \max_i \sum_j ||f(x_i)-f(x_j)||^2}$ - $1 \le i, j \le \textrm{NumClusters}$
        - This avoids a large fraction of computing non-discriminative shapelet candidates
        - Finally, adopt a classical classifier (e.g., SVM)

![21li](img/21li.png)

## Multivar Multi-layer NN: [Medico, 2021](https://doi.org/10.1007/s11634-021-00437-8) -- **Shapelets as weights, 3 losses**
- **Multi-layer neural network**: BatchNorm, *Distance*, BatchNorm, *Linear(s) + ReLU*, BatchNorm, Dropout, *Softmax*
    - Distance layer: Shapelets *themselves* are the weights! 
    - Initialize $\textbf{S}$ using KMeans on each channel
- **Learning objective**: three components:
    - $\textrm{Loss}_\textrm{Classification} = -\sum_i y_i \log \hat{y}_i$
    - $\textrm{Loss}_\textrm{Redundancy} = \sum_j \frac{\sigma(G_j)}{k}$, $\sigma(G_j) = \frac{1}{1+e^{-\theta G_j}}$ - $\textbf{G} \in \Bbb{R}^k$ (gating params.), $\theta$ $\gg 1$
        - Suppress **redundant** shapelets: sigmoid forces $\textbf{G}$ to be *binary*
    - $\textrm{Loss}_\textrm{Correlation} = \max_{i, j} |\textrm{CorrMat}(\textbf{M} \cdot \textrm{diag}(\sigma(G_j)))_{ij}|$, $M_{ij} = \textrm{dist}(s_i, v_j)$
        - Suppress **similar** shapelets: now $\textbf{G}$ is part of the differentiable computing graph! 
    - $L_\textrm{MLNN} = (1-\alpha) \textrm{Loss}_\textrm{Cls} + \alpha (\textrm{Loss}_\textrm{Rdn} + \textrm{Loss}_\textrm{Corr})$, $\alpha \in [0, \frac{1}{2}]$ (cosine cyclic scheduling: H - discard, L - refine)
        - Learnables: weights $\textbf{W}$ (Linear & Softmax, cls), shapelets $\textbf{S}$ & gatings $\textbf{G}$ (Distance, rdn & corr)
        - This objective selects *smaller* sets of *uncorrelated* shapelets (no selection of hyperparams. like $k$ and $m$)

![21medico](img/21medico.png)

## Time2Vec: [Kazemi, 2019](https://doi.org/10.48550/arXiv.1907.05321)
- Key properties
    - Capturing (non-)**periodic** patterns
    - Being **invarient** to time scaling
    - Being **simple** enough for model combination
- $\textrm{Time2Vec}(t)[i] = \mathcal{F}(\omega_i t + \tau_i)$ - $\mathcal{F}$ is periodic activation, $\omega_i$ & $\tau_i$ are learnable params.

<br>

# Digest: Possible Technological Paths

## Previously……
- [Time2Graph](https://doi.org/10.1609/AAAI.v34i04.5769) as a Finite State Machine
- "Possibilities" in the state transition diagram: from $t$-$v$ distance to learned embedding vectors

## The journey of research
- First thoughts: unsupervised framework to generate shapelets without labels?
    - GAN, variational encoder, etc. 
    - Hardly moves away from the $t$-$v$ distance
    - Generative models only aim to map similar inputs to similar outputs (makes no difference in clustering)
- After hitting the dead end: finding *diverse* shapelets
    - [ADSN](https://doi.org/10.1609/AAAI.v34i04.5948): generative + dynamic model
        - Exigency: learned shapelets may be very different from the subsequences
        - Methodology: sample-wise ("dynamic") shapelets - convolve the subsequences, discriminate with GAN
        - Issue: how can shapelets generated "on the fly" represent *global* features? 
    - [ShapeNet](https://ojs.aaai.org/index.php/AAAI/article/view/17018): moves away from the traditional "classification" criterion
        - Exigency: in multivar. TS, it's hard to compare shapelets with various lengths
        - Methodology: embed the subsequences into a unified space, select *large and isolated* clusters
        - Issue: why don't we directly cluster the subsequences? 

## Useful Components
- **Category- & sample-level ("dynamic") shapelets**: learned (global) convolution kernels
    - Does it make sense to get rid of global *shapelets*? We don't want to compare a series to itself
- **Generative adversarial network**: update classifier -> disriminator (multi-layer conv-nn) -> shapelets consecutively
    - Coding could be massive…… and GAN requires a brand new field of background knowledge
- **Dilated causal CNN**: no future values, with blanks. *((conv, norm, relu) * 2, res) * 3, maxpool, linear, softmax*
    - It's considerate to be *causal*, yet it (1) is computationally expensive and (2) simply "tightens" clusters
- **Multi-layer NN**: shapelets themselves as weights; motif-based init. *linear, relu, dropout, softmax*
    - Seems old-fashioned, but a similar MLP could be useful; think twice on param. initialization
- **All kinds of losses**
    - Classification (cross entropy, **encourage accuracy**) - this is a must
    - Adversarial ($v$-$t$ similarity, **encourage "beautiful" shapelets**) - see above for the GAN issues…… :-L
    - Diversity / Correlation ($v$-$v$ similarity, **encourage diverse shapelets**) - redundant with clustering selection
    - Redundancy (w/gating params., **encourage less shapelets**) - auto-tuning of "$k$"-means
    - Triplet ($A$-$P$-$N$, **encourage clear-cut clusters**) - picking good samples of $P$ and $N$ could be an issue

## Improved algorithm outline
- Unsupervised labelling: pick the "anchor" shapelets with KMeans, calculate $t$-$v$ distances as labels
- **Unsupervised step**: *contrastively* train an embedding network $\mathcal{U}$ to extract $t$'s features
- Supervised labelling: feed $t$ and $v$ into $\mathcal{U}$, use the E.D. of *output vectors* as the "distance" labels
- **Supervised step**: concatenate an MLP $\mathcal{S}$ after the unsupervised model, use the labels ^ for fine-tuning
- The model is then ready. For each $s$, transfer its $t$s into possibility vectors and contruct the FSM Diagram

## More on picking shapelet candidates
- Randomly pick among the segments
- **KMeans** algorithm: classify $t$s into clusters, choose the center of each
    - Clustering encourages **representative** shapelets, but **diversity** requires more thoughts
    - A mature algrm. for speeding up (in $O(n)$ time)
    - [ShapeNet](https://ojs.aaai.org/index.php/AAAI/article/view/17018) shows the plausibility of selection via clustering
        - ^ modified ver.: choose the centroids in the *nn-learned* embedding space (WHAT FOR?)

## Other possibilities
- Graph Learning Method: **GIN** (Graph Isomorphism Network)
- Make use of different time scales: 1 month, 3 months, 6 months
- Use deep learning to learn the shapelets (as proposed above)
