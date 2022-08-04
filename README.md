# Unsupervised Representation Learning from Pathology Images with Multi-directional Contrastive Predictive Coding
**Jacob Carse¹, Frank Carey² and Stephen McKenna¹**

¹CVIP, School of Science and Engineering, University of Dundee, Dundee, Scotland, UK\
²Department of Pathology, Ninewells Hospital and Medical School, Dundee, Scotland, UK


Digital pathology tasks have benefited greatly from modern deep learning algorithms. However, their need for large quantities of annotated data has been identified as a key challenge. This need for data can be countered by using unsupervised learning in situations where data are abundant but access to annotations is limited. Feature representations learned from unannotated data using contrastive predictive coding (CPC) have been shown to enable classifiers to obtain state of the art performance from relatively small amounts of annotated computer vision data. We present a modification to the CPC framework for use with digital pathology patches. This is achieved by introducing an alternative mask for building the latent context and using a multi-directional PixelCNN autoregressor. To demonstrate our proposed method we learn feature representations from the Patch Camelyon histology dataset. We show that our proposed modification can yield improved deep classification of histology patches.
