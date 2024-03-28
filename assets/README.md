This is a complete list of all plots in the repo as well as the scripts and data used to generate them.

figure					 | plot							 | script							 | data
-------------------------|-------------------------------|-----------------------------------|-------------------------------------------------------------------
fig:fit_alginate		 | fit_alginate_eta.pdf			 | fig:fit_alginate.py				 | Daten_SOP_BioPrint_Alginat_PH176/SOP_Bioprint_Probe_{i}.csv i=2-22
						 | fit_alginate_N.pdf			 | fig:fit_alginate.py				 | Daten_SOP_BioPrint_Alginat_PH176/SOP_Bioprint_Probe_{i}.csv i=2-22
fig:fit_mc				 | fit_mc_eta.pdf				 | fig:fit_mc.py					 | methylcellulose
						 | fit_mc_N.pdf					 | fig:fit_mc.py					 | methylcellulose
fig:2Dvel				 | 2DAlginateU.pdf				 | fig:2DAlginate.py				 | novel_2024-01-15/0/
						 | 2DParameterStudyUWorst.pdf	 | fig:2DParameterStudy.py			 | 2DParameterStudy/7/
fig:3Dvel				 | 3DAlginateU.pdf				 | fig:3DAlginate.py				 | novel_2024-01-15/1/
						 | 3DParameterStudyUWorst.pdf	 | fig:3DParameterStudy.py			 | 3DParameterStudy/7/
fig:malaspinas2010		 | malaspinas2010.pdf			 | fig:malaspinas2010.py			 | malaspinas2010
fig:Nozzle				 | Nozzle_v.pdf					 | fig:Nozzle.py					 | novel_2024-03-26/2/
						 | Nozzle_tau_vM.pdf			 | fig:Nozzle.py					 | novel_2024-03-26/2/
						 | Nozzle_tau_12.pdf			 | fig:Nozzle.py					 | novel_2024-03-26/2/
fig:RT-DC				 | RT-DC_v.pdf					 | fig:RT-DC.py						 | RT-DC/0/
						 | RT-DC_tau_vM.pdf				 | fig:RT-DC.py						 | RT-DC/0/
						 | RT-DC_tau_12.pdf				 | fig:RT-DC.py						 | RT-DC/0/
fig:cellInShear_vM		 | cellInShear_sigma_vM.pdf		 | fig:CellInShear.py				 | cellInShear_2024-01-23/0/
fig:wiParameterStudy	 | wiParameterStudy.pdf			 | fig:wiParameterStudy.py			 | wiParameterStudy/{i} i=0-14
						 | suffleParameterStudy.pdf		 | fig:shuffleParameterStudy.py		 | SuffleParameterStudy/{i} i=0-10;20-30
fig:2DTimeEvolution		 | 2DAlginate_timeEvolution.pdf	 | fig:iDAlginate_timeEvolution.py	 | novel_2024-01-15/0/
						 | 2DMC_timeEvolution.pdf		 | fig:iDMC_timeEvolution.py		 | 2DParameterStudy/7/
fig:2DvelErr			 | 2DAlginateErr.pdf 			 | fig:2DAlginate.py				 | novel_2024-01-15/0/
						 | 2DParameterStudyErrWorst.pdf	 | fig:2DParameterStudy.py			 | 2DParameterStudy/7/
fig:pressureErr			 | 2DParameterStudyErr.pdf		 | fig:2DParameterStudy.py			 | 2DParameterStudy/{i}/ i=0-14
						 | 3DParameterStudyErr.pdf		 | fig:3DParameterStudy.py			 | 3DParameterStudy/{i}/ i=0-11
fig:3DTimeEvolution		 | 3DAlginate_timeEvolution.pdf	 | fig:iDAlginate_timeEvolution.py	 | novel_2024-01-15/1/
						 | 3DMC_timeEvolution.pdf		 | fig:iDMC_timeEvolution.py		 | 3DParameterStudy/7/
fig:3DvelErr			 | 3DAlginateErr.pdf			 | fig:3DAlginate.py				 | novel_2024-01-15/1/
						 | 3DParameterStudyErrWorst.pdf	 | fig:3DParameterStudy.py			 | 3DParameterStudy/7/
fig:ShuffleInPoiseuille	 | ShuffleInPoiseuille.pdf		 | fig:ShuffleInPoiseuille.py		 | ShuffleInPoiseuille/{i} i=0,1,2b,3,4
fig:Nozzle/RT-DC_eta	 | Nozzle_eta.pdf				 | fig:Nozzle.py					 | novel_2024-03-26/2/
						 | RT-DC_eta.pdf				 | fig:RT-DC.py						 | RT-DC/0/

The following plots are present in the repo but where not used in the paper.<br>
The scripts without plots either where deprecated, did additional analysis not used in the paper or did prep work for sims.<br>

plot								 | script					 | data
-------------------------------------|---------------------------|--------------------------
2DParameterStudyErr{i}.pdf			 | fig:2DParameterStudy.py	 | 2DParameterStudy/{i}/ i=0-14
2DParameterStudyU{i}.pdf			 | fig:2DParameterStudy.py	 | 2DParameterStudy/{i}/ i=0-14
3DParameterStudyErr{i}.pdf			 | fig:3DParameterStudy.py	 | 3DParameterStudy/{i}/ i=0-11
3DParameterStudyU{i}.pdf			 | fig:3DParameterStudy.py	 | 3DParameterStudy/{i}/ i=0-11
cellInShear_v.pdf					 | fig:CellInShear.py		 | cellInShear_2024-01-23/0/
RT-DC_eta_cut.pdf					 | fig:RT-DC.py				 | RT-DC/0/
RT-DC_tau_12_graphical_abstract.pdf	 | fig:RT-DC.py				 | RT-DC/0/
									 | cellInHighShearRoscoe.py	 | cellInShear_2024-02-05/0/
									 | cellInShearRoscoe.py		 | cellInShear_2024-01-23/0/
									 | CY.py					 |
									 | fig:ptt_eta.py			 |
									 | fig:ptt_N.py				 |
									 | fig:RT-DC_CY.py			 | RT-DC/1/
									 | flowRate.py				 |
									 | inputForNozzle.py		 | Nozzle/{i}/ i=0,1
									 | prepare2DAlginate.py		 |
									 | prepare3DAlginate.py		 |
									 | prepareCY.py				 |

In `python/fluidx3d` general analysis tools can be found. These where devoloped during the writing of this paper and as such are not used in all scripts.
The setups used to generate the data can be found in setups using the same name as the python scripts evaluating the output.
