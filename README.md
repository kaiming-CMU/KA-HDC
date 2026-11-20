# KA-HDC

**Kernel Alignment with Heat Diffusion Consensus for Multi-View Spatial Transcriptomics Clustering**

KA-HDC identifies spatial domains from a spot-by-gene expression matrix and the tissue coordinates of each spot. It learns from two graphs over the same spots. One connects spots that are close in the tissue, and the other connects spots with similar expression profiles. The output is a spatial domain label for each spot.

## Code availability

The manuscript is under review. During peer review, we are releasing selected components of KA-HDC. We will release the full implementation in this repository upon acceptance.

## Framework

[![KA-HDC framework showing the two graph encoders, cross-view kernel alignment, heat diffusion consensus, and diffusion-guided contrastive learning](figures/framework.png)](figures/framework.pdf)

[View the full-resolution figure (PDF)](figures/framework.pdf)

1. **Encode the two graph views.** The spatial graph connects spots within a chosen radius. The expression graph connects each spot to its nearest neighbors by cosine similarity. A separate graph convolutional network encodes each view, using gene expression as the input for both.
2. **Align within-view similarities (panel A).** KA-HDC computes a Gaussian kernel over the spot embeddings in each view. An objective based on the Hilbert–Schmidt independence criterion (HSIC) encourages the two views to agree on which spots are similar.
3. **Build a diffusion consensus (panel B).** KA-HDC averages the two kernel matrices to form a shared similarity graph and applies heat diffusion to it. Diffusion time controls how far information spreads. Each row of the normalized heat kernel gives one spot's affinity to every spot in the section.
4. **Use diffusion affinities as contrastive targets (panel C).** KA-HDC trains each view to predict these affinities from its similarities to spots in the other view. Training combines this bidirectional contrastive loss with kernel alignment and graph reconstruction.

After training, KA-HDC averages the two embeddings for each spot and applies k-means to assign spatial domains.

## Evaluation

We evaluated KA-HDC on 12 human dorsolateral prefrontal cortex (DLPFC) slices, one human breast cancer section, and one anterior mouse brain section. KA-HDC achieved the highest adjusted Rand index (ARI) among the evaluated methods on **10 of 14 sections**, including **9 of 12 DLPFC slices** and the mouse brain section. On DLPFC slices 151669 and 151670, KA-HDC achieved ARIs of 0.74 and 0.79, respectively. The best baseline scores on those slices were 0.57 and 0.54.
