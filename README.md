# Kinase-Binding-Encoder-Work-in-Progress-
Ligand and protein encoder training pipeline with ChEMBL and KLIFS databases to create encoders that segregate in latent space by binding and by binding mechanism

Encoder for ligands that bind to kinases, trained in two phases:
Phase 1 - pretraining: graph neural network (GNN) ligand encoder and ESM-2 protein encoder trained jointly with BCE objective on ChEMBL dataset filtered for kinases only
  Phase 1's goal is to separate embeddings in the latent space based on whether or not ligands bind to similar proteins 
Phase 2 - finetuning: ligand encoder finetuned with KLIFs kinase database using triplet loss
  Phase 2's goal is to move embeddings of ligands sharing a binding mode (allosteric, covalent, etc) closer together in the latent space, while those that differ are pushed further apart

This project is a work-in-progress!

Details: 
-Ligand encoder: GNN representation (GINEConv)
  node features: atomic number, formal charge, hydrogens, degree, hybridization, aromatic (binary)
  edge features (bonds between atoms): bond type (single,double..),stereochemistry, ring (binary), conjugated (binary)
-Protein encoder: ESM-2 model as initial representation of AA sequence for proteins 
  Both Phase 1 and Phase 2 utilize full AA sequence for kinases currently
  (KLIFs dataset cleaner file also computes the 85-AA sequence of the binding pocket, but this is not currently being trained on in phase 2)

-Phase 1 labels currently using hard thresholds based on ChEMBL dataset values of Ki, Kd, and IC50, as metrics of binding affinity for molecules. This stands to be adjusted as project develops to some sort of trained classifier to assign labels, but as of now metrics are binary



Status/Current limitations:
- Running on smaller dataset hard-coding to CPU currently to set up personal computer to be able to handle larger datasets and GPU! 
- Phase 1/Phase 2 target list is not exact: the KLIFs database provides detailed information about the mechanism of binding, which is used to further distinguish embeddings in the latenet space. However, the KLIFs database is smaller than the ChEMBL dataset, and the kinases present in either list are not precisely aligned
- Phase 2's finetuning phase adjsuts for mechanism by which a ligand binds, but the LigandEncoder is blind to the kinase it's paired with, so if a ligand binds to multiple kinases with multiple mechanisms, the triplet loss can pull an embedding in conflciting directions depending in the triplet. The embedding after going through pretraining and finetuning represents ligand's dominant mechanistic binding behavior, not its behavior against all kinases it binds with 


  
