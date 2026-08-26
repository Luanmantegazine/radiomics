
 mri_convert /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/RAW/5/file.dcm /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Thu Jan 01 00:00:00 CST 1970

 cp /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/orig/001.mgz /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/rawavg.mgz 


 mri_convert /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/rawavg.mgz /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/transforms/talairach.xfm /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/orig.mgz /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Thu Jan 01 00:00:00 CST 1970

 mri_nu_correct.mni --n 1 --proto-iters 1000 --distance 50 --no-rescale --i orig.mgz --o orig_nu.mgz 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 


 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

#--------------------------------------------
#@# Talairach Failure Detection Thu Jan 01 00:00:00 CST 1970

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /nrgpackages/tools/freesurfer53-patch/bin/extract_talairach_avi_QA.awk /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/transforms/talairach_avi.log 


 tal_QC_AZS /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Thu Jan 01 00:00:00 CST 1970

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 


 mri_add_xform_to_header -c /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Thu Jan 01 00:00:00 CST 1970

 mri_normalize -g 1 -mprage nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Thu Jan 01 00:00:00 CST 1970

 mri_em_register -skull nu.mgz /nrgpackages/tools/freesurfer53-patch/average/RB_all_withskull_2008-03-26.gca transforms/talairach_with_skull.lta 


 mri_watershed -T1 -brain_atlas /nrgpackages/tools/freesurfer53-patch/average/RB_all_withskull_2008-03-26.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


 cp brainmask.auto.mgz brainmask.mgz 

#-------------------------------------
#@# EM Registration Thu Jan 01 00:00:00 CST 1970

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /nrgpackages/tools/freesurfer53-patch/average/RB_all_2008-03-26.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Thu Jan 01 00:00:00 CST 1970

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /nrgpackages/tools/freesurfer53-patch/average/RB_all_2008-03-26.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Thu Jan 01 00:00:00 CST 1970

 mri_ca_register -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /nrgpackages/tools/freesurfer53-patch/average/RB_all_2008-03-26.gca transforms/talairach.m3z 

#--------------------------------------
#@# Remove Neck Thu Jan 01 00:00:00 CST 1970

 mri_remove_neck -radius 25 nu.mgz transforms/talairach.m3z /nrgpackages/tools/freesurfer53-patch/average/RB_all_2008-03-26.gca nu_noneck.mgz 

#--------------------------------------
#@# SkullLTA Thu Jan 01 00:00:00 CST 1970

 mri_em_register -skull -t transforms/talairach.lta nu_noneck.mgz /nrgpackages/tools/freesurfer53-patch/average/RB_all_withskull_2008-03-26.gca transforms/talairach_with_skull_2.lta 

#--------------------------------------
#@# SubCort Seg Thu Jan 01 00:00:00 CST 1970

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /nrgpackages/tools/freesurfer53-patch/average/RB_all_2008-03-26.gca aseg.auto_noCCseg.mgz 


 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri/transforms/cc_up.lta OAS30001_MR_d0757 

#--------------------------------------
#@# Merge ASeg Thu Jan 01 00:00:00 CST 1970

 cp aseg.auto.mgz aseg.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Thu Jan 01 00:00:00 CST 1970

 mri_normalize -mprage -aseg aseg.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Thu Jan 01 00:00:00 CST 1970

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Thu Jan 01 00:00:00 CST 1970

 mri_segment -mprage brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Thu Jan 01 00:00:00 CST 1970

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.auto_noCCseg.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Thu Jan 01 00:00:00 CST 1970

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Thu Jan 01 00:00:00 CST 1970

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Thu Jan 01 00:00:00 CST 1970

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Thu Jan 01 00:00:00 CST 1970

 mris_sphere -q -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# Fix Topology lh Thu Jan 01 00:00:00 CST 1970

 cp ../surf/lh.orig.nofix ../surf/lh.orig 


 cp ../surf/lh.inflated.nofix ../surf/lh.inflated 


 mris_fix_topology -mgz -sphere qsphere.nofix -ga -seed 1234 OAS30001_MR_d0757 lh 


 mris_euler_number ../surf/lh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm ../surf/lh.inflated 

#--------------------------------------------
#@# Make White Surf lh Thu Jan 01 00:00:00 CST 1970

 mris_make_surfaces -noaparc -whiteonly -mgz -T1 brain.finalsurfs OAS30001_MR_d0757 lh 

#--------------------------------------------
#@# Smooth2 lh Thu Jan 01 00:00:00 CST 1970

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white ../surf/lh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Thu Jan 01 00:00:00 CST 1970

 mris_inflate ../surf/lh.smoothwm ../surf/lh.inflated 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 ../surf/lh.inflated 


#-----------------------------------------
#@# Curvature Stats lh Thu Jan 01 00:00:00 CST 1970

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm OAS30001_MR_d0757 lh curv sulc 

#--------------------------------------------
#@# Sphere lh Thu Jan 01 00:00:00 CST 1970

 mris_sphere -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Surf Reg lh Thu Jan 01 00:00:00 CST 1970

 mris_register -curv ../surf/lh.sphere /nrgpackages/tools/freesurfer53-patch/average/lh.average.curvature.filled.buckner40.tif ../surf/lh.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Thu Jan 01 00:00:00 CST 1970

 mris_jacobian ../surf/lh.white ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Thu Jan 01 00:00:00 CST 1970

 mrisp_paint -a 5 /nrgpackages/tools/freesurfer53-patch/average/lh.average.curvature.filled.buckner40.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 lh ../surf/lh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/lh.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs ../label/lh.aparc.annot 

#--------------------------------------------
#@# Make Pial Surf lh Thu Jan 01 00:00:00 CST 1970

 mris_make_surfaces -white NOWRITE -mgz -T1 brain.finalsurfs OAS30001_MR_d0757 lh 

#--------------------------------------------
#@# Surf Volume lh Thu Jan 01 00:00:00 CST 1970

 mris_calc -o lh.area.mid lh.area add lh.area.pial 


 mris_calc -o lh.area.mid lh.area.mid div 2 


 mris_calc -o lh.volume lh.area.mid mul lh.thickness 

#--------------------------------------------
#@# Tessellate rh Thu Jan 01 00:00:00 CST 1970

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 rh Thu Jan 01 00:00:00 CST 1970

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 rh Thu Jan 01 00:00:00 CST 1970

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere rh Thu Jan 01 00:00:00 CST 1970

 mris_sphere -q -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#--------------------------------------------
#@# Fix Topology rh Thu Jan 01 00:00:00 CST 1970

 cp ../surf/rh.orig.nofix ../surf/rh.orig 


 cp ../surf/rh.inflated.nofix ../surf/rh.inflated 


 mris_fix_topology -mgz -sphere qsphere.nofix -ga -seed 1234 OAS30001_MR_d0757 rh 


 mris_euler_number ../surf/rh.orig 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm ../surf/rh.inflated 

#--------------------------------------------
#@# Make White Surf rh Thu Jan 01 00:00:00 CST 1970

 mris_make_surfaces -noaparc -whiteonly -mgz -T1 brain.finalsurfs OAS30001_MR_d0757 rh 

#--------------------------------------------
#@# Smooth2 rh Thu Jan 01 00:00:00 CST 1970

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 rh Thu Jan 01 00:00:00 CST 1970

 mris_inflate ../surf/rh.smoothwm ../surf/rh.inflated 


 mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 ../surf/rh.inflated 


#-----------------------------------------
#@# Curvature Stats rh Thu Jan 01 00:00:00 CST 1970

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm OAS30001_MR_d0757 rh curv sulc 

#--------------------------------------------
#@# Sphere rh Thu Jan 01 00:00:00 CST 1970

 mris_sphere -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg rh Thu Jan 01 00:00:00 CST 1970

 mris_register -curv ../surf/rh.sphere /nrgpackages/tools/freesurfer53-patch/average/rh.average.curvature.filled.buckner40.tif ../surf/rh.sphere.reg 

#--------------------------------------------
#@# Jacobian white rh Thu Jan 01 00:00:00 CST 1970

 mris_jacobian ../surf/rh.white ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv rh Thu Jan 01 00:00:00 CST 1970

 mrisp_paint -a 5 /nrgpackages/tools/freesurfer53-patch/average/rh.average.curvature.filled.buckner40.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc rh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 rh ../surf/rh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/rh.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# Make Pial Surf rh Thu Jan 01 00:00:00 CST 1970

 mris_make_surfaces -white NOWRITE -mgz -T1 brain.finalsurfs OAS30001_MR_d0757 rh 

#--------------------------------------------
#@# Surf Volume rh Thu Jan 01 00:00:00 CST 1970

 mris_calc -o rh.area.mid rh.area add rh.area.pial 


 mris_calc -o rh.area.mid rh.area.mid div 2 


 mris_calc -o rh.volume rh.area.mid mul rh.thickness 

#-----------------------------------------
#@# Parcellation Stats lh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab OAS30001_MR_d0757 lh white 

#-----------------------------------------
#@# Cortical Parc 2 lh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 lh ../surf/lh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/lh.destrieux.simple.2009-07-29.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Parcellation Stats 2 lh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab OAS30001_MR_d0757 lh white 

#-----------------------------------------
#@# Cortical Parc 3 lh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 lh ../surf/lh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/lh.DKTatlas40.gcs ../label/lh.aparc.DKTatlas40.annot 

#-----------------------------------------
#@# Parcellation Stats 3 lh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas40.stats -b -a ../label/lh.aparc.DKTatlas40.annot -c ../label/aparc.annot.DKTatlas40.ctab OAS30001_MR_d0757 lh white 

#-----------------------------------------
#@# WM/GM Contrast lh Thu Jan 01 00:00:00 CST 1970

 pctsurfcon --s OAS30001_MR_d0757 --lh-only 

#-----------------------------------------
#@# Parcellation Stats rh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab OAS30001_MR_d0757 rh white 

#-----------------------------------------
#@# Cortical Parc 2 rh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 rh ../surf/rh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/rh.destrieux.simple.2009-07-29.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Parcellation Stats 2 rh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab OAS30001_MR_d0757 rh white 

#-----------------------------------------
#@# Cortical Parc 3 rh Thu Jan 01 00:00:00 CST 1970

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.mgz -seed 1234 OAS30001_MR_d0757 rh ../surf/rh.sphere.reg /nrgpackages/tools/freesurfer53-patch/average/rh.DKTatlas40.gcs ../label/rh.aparc.DKTatlas40.annot 

#-----------------------------------------
#@# Parcellation Stats 3 rh Thu Jan 01 00:00:00 CST 1970

 mris_anatomical_stats -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas40.stats -b -a ../label/rh.aparc.DKTatlas40.annot -c ../label/aparc.annot.DKTatlas40.ctab OAS30001_MR_d0757 rh white 

#-----------------------------------------
#@# WM/GM Contrast rh Thu Jan 01 00:00:00 CST 1970

 pctsurfcon --s OAS30001_MR_d0757 --rh-only 

#--------------------------------------------
#@# Cortical ribbon mask Thu Jan 01 00:00:00 CST 1970

 mris_volmask --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon OAS30001_MR_d0757 

#--------------------------------------------
#@# ASeg Stats Thu Jan 01 00:00:00 CST 1970

 mri_segstats --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /nrgpackages/tools/freesurfer53-patch/ASegStatsLUT.txt --subject OAS30001_MR_d0757 

#-----------------------------------------
#@# AParc-to-ASeg Thu Jan 01 00:00:00 CST 1970

 mri_aparc2aseg --s OAS30001_MR_d0757 --volmask 


 mri_aparc2aseg --s OAS30001_MR_d0757 --volmask --a2009s 

#-----------------------------------------
#@# WMParc Thu Jan 01 00:00:00 CST 1970

 mri_aparc2aseg --s OAS30001_MR_d0757 --labelwm --hypo-as-wm --rip-unknown --volmask --o mri/wmparc.mgz --ctxseg aparc+aseg.mgz 


 mri_segstats --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject OAS30001_MR_d0757 --surf-wm-vol --ctab /nrgpackages/tools/freesurfer53-patch/WMParcStatsLUT.txt --etiv 

#--------------------------------------------
#@# Qdec Cache lh thickness fsaverage Thu Jan 01 00:00:00 CST 1970
INFO: fsaverage subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to fsaverage subject...

 cd /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED; ln -s /nrgpackages/tools/freesurfer53-patch/subjects/fsaverage; cd - 


 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas thickness --target fsaverage --out lh.thickness.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh thickness fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.thickness.fsaverage.mgh --tval lh.thickness.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas area --target fsaverage --out lh.area.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh area fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.area.fsaverage.mgh --tval lh.area.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas area.pial --target fsaverage --out lh.area.pial.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh area.pial fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.area.pial.fsaverage.mgh --tval lh.area.pial.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas volume --target fsaverage --out lh.volume.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh volume fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.volume.fsaverage.mgh --tval lh.volume.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas curv --target fsaverage --out lh.curv.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh curv fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.curv.fsaverage.mgh --tval lh.curv.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas sulc --target fsaverage --out lh.sulc.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh sulc fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.sulc.fsaverage.mgh --tval lh.sulc.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh white.K fsaverage Thu Jan 01 00:00:00 CST 1970
INFO: File lh.white.K does not exist!
Skipping creation of smoothed data for lh.white.K
#--------------------------------------------
#@# Qdec Cache lh white.H fsaverage Thu Jan 01 00:00:00 CST 1970
INFO: File lh.white.H does not exist!
Skipping creation of smoothed data for lh.white.H
#--------------------------------------------
#@# Qdec Cache lh jacobian_white fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas jacobian_white --target fsaverage --out lh.jacobian_white.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh jacobian_white fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.jacobian_white.fsaverage.mgh --tval lh.jacobian_white.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi lh --meas w-g.pct.mgh --target fsaverage --out lh.w-g.pct.mgh.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 0 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 5 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 10 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 15 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 20 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache lh w-g.pct.mgh fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi lh --fwhm 25 --sval lh.w-g.pct.mgh.fsaverage.mgh --tval lh.w-g.pct.mgh.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas thickness --target fsaverage --out rh.thickness.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh thickness fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.thickness.fsaverage.mgh --tval rh.thickness.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas area --target fsaverage --out rh.area.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh area fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.area.fsaverage.mgh --tval rh.area.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas area.pial --target fsaverage --out rh.area.pial.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh area.pial fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.area.pial.fsaverage.mgh --tval rh.area.pial.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas volume --target fsaverage --out rh.volume.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh volume fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.volume.fsaverage.mgh --tval rh.volume.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas curv --target fsaverage --out rh.curv.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh curv fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.curv.fsaverage.mgh --tval rh.curv.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas sulc --target fsaverage --out rh.sulc.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh sulc fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.sulc.fsaverage.mgh --tval rh.sulc.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh white.K fsaverage Thu Jan 01 00:00:00 CST 1970
INFO: File rh.white.K does not exist!
Skipping creation of smoothed data for rh.white.K
#--------------------------------------------
#@# Qdec Cache rh white.H fsaverage Thu Jan 01 00:00:00 CST 1970
INFO: File rh.white.H does not exist!
Skipping creation of smoothed data for rh.white.H
#--------------------------------------------
#@# Qdec Cache rh jacobian_white fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas jacobian_white --target fsaverage --out rh.jacobian_white.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh jacobian_white fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.jacobian_white.fsaverage.mgh --tval rh.jacobian_white.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fsaverage Thu Jan 01 00:00:00 CST 1970

 mris_preproc --s OAS30001_MR_d0757 --hemi rh --meas w-g.pct.mgh --target fsaverage --out rh.w-g.pct.mgh.fsaverage.mgh 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm0 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 0 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm0.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm5 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 5 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm5.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm10 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 10 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm10.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm15 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 15 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm15.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm20 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 20 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm20.fsaverage.mgh --cortex 

#--------------------------------------------
#@# Qdec Cache rh w-g.pct.mgh fwhm25 fsaverage Thu Jan 01 00:00:00 CST 1970

 mri_surf2surf --s fsaverage --hemi rh --fwhm 25 --sval rh.w-g.pct.mgh.fsaverage.mgh --tval rh.w-g.pct.mgh.fwhm25.fsaverage.mgh --cortex 

#--------------------------------------------
#@# BA Labels lh Thu Jan 01 00:00:00 CST 1970

 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA1.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA1.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA2.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA2.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA3a.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA3a.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA3b.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA3b.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA4a.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA4a.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA4p.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA4p.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA6.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA6.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA44.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA44.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA45.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA45.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.V1.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.V1.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.V2.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.V2.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.MT.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.MT.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.perirhinal.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.perirhinal.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA1.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA1.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA2.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA2.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA3a.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA3a.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA3b.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA3b.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA4a.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA4a.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA4p.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA4p.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA6.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA6.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA44.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA44.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.BA45.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.BA45.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.V1.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.V1.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.V2.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.V2.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/lh.MT.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./lh.MT.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s OAS30001_MR_d0757 --hemi lh --ctab /nrgpackages/tools/freesurfer53-patch/average/colortable_BA.txt --l lh.BA1.label --l lh.BA2.label --l lh.BA3a.label --l lh.BA3b.label --l lh.BA4a.label --l lh.BA4p.label --l lh.BA6.label --l lh.BA44.label --l lh.BA45.label --l lh.V1.label --l lh.V2.label --l lh.MT.label --l lh.perirhinal.label --a BA --maxstatwinner --noverbose 


 mris_label2annot --s OAS30001_MR_d0757 --hemi lh --ctab /nrgpackages/tools/freesurfer53-patch/average/colortable_BA.txt --l lh.BA1.thresh.label --l lh.BA2.thresh.label --l lh.BA3a.thresh.label --l lh.BA3b.thresh.label --l lh.BA4a.thresh.label --l lh.BA4p.thresh.label --l lh.BA6.thresh.label --l lh.BA44.thresh.label --l lh.BA45.thresh.label --l lh.V1.thresh.label --l lh.V2.thresh.label --l lh.MT.thresh.label --a BA.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -mgz -f ../stats/lh.BA.stats -b -a ./lh.BA.annot -c ./BA.ctab OAS30001_MR_d0757 lh white 


 mris_anatomical_stats -mgz -f ../stats/lh.BA.thresh.stats -b -a ./lh.BA.thresh.annot -c ./BA.thresh.ctab OAS30001_MR_d0757 lh white 

#--------------------------------------------
#@# BA Labels rh Thu Jan 01 00:00:00 CST 1970

 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA1.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA1.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA2.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA2.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA3a.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA3a.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA3b.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA3b.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA4a.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA4a.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA4p.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA4p.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA6.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA6.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA44.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA44.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA45.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA45.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.V1.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.V1.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.V2.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.V2.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.MT.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.MT.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.perirhinal.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.perirhinal.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA1.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA1.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA2.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA2.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA3a.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA3a.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA3b.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA3b.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA4a.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA4a.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA4p.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA4p.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA6.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA6.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA44.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA44.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.BA45.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.BA45.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.V1.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.V1.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.V2.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.V2.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/fsaverage/label/rh.MT.thresh.label --trgsubject OAS30001_MR_d0757 --trglabel ./rh.MT.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s OAS30001_MR_d0757 --hemi rh --ctab /nrgpackages/tools/freesurfer53-patch/average/colortable_BA.txt --l rh.BA1.label --l rh.BA2.label --l rh.BA3a.label --l rh.BA3b.label --l rh.BA4a.label --l rh.BA4p.label --l rh.BA6.label --l rh.BA44.label --l rh.BA45.label --l rh.V1.label --l rh.V2.label --l rh.MT.label --l rh.perirhinal.label --a BA --maxstatwinner --noverbose 


 mris_label2annot --s OAS30001_MR_d0757 --hemi rh --ctab /nrgpackages/tools/freesurfer53-patch/average/colortable_BA.txt --l rh.BA1.thresh.label --l rh.BA2.thresh.label --l rh.BA3a.thresh.label --l rh.BA3b.thresh.label --l rh.BA4a.thresh.label --l rh.BA4p.thresh.label --l rh.BA6.thresh.label --l rh.BA44.thresh.label --l rh.BA45.thresh.label --l rh.V1.thresh.label --l rh.V2.thresh.label --l rh.MT.thresh.label --a BA.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -mgz -f ../stats/rh.BA.stats -b -a ./rh.BA.annot -c ./BA.ctab OAS30001_MR_d0757 rh white 


 mris_anatomical_stats -mgz -f ../stats/rh.BA.thresh.stats -b -a ./rh.BA.thresh.annot -c ./BA.thresh.ctab OAS30001_MR_d0757 rh white 

#--------------------------------------------
#@# Ex-vivo Entorhinal Cortex Label lh Thu Jan 01 00:00:00 CST 1970
INFO: lh.EC_average subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to lh.EC_average subject...

 cd /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED; ln -s /nrgpackages/tools/freesurfer53-patch/subjects/lh.EC_average; cd - 


 mris_spherical_average -erode 1 -orig white -t 0.4 -o OAS30001_MR_d0757 label lh.entorhinal lh sphere.reg lh.EC_average lh.entorhinal_exvivo.label 


 mris_anatomical_stats -mgz -f ../stats/lh.entorhinal_exvivo.stats -b -l ./lh.entorhinal_exvivo.label OAS30001_MR_d0757 lh white 

#--------------------------------------------
#@# Ex-vivo Entorhinal Cortex Label rh Thu Jan 01 00:00:00 CST 1970
INFO: rh.EC_average subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to rh.EC_average subject...

 cd /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED; ln -s /nrgpackages/tools/freesurfer53-patch/subjects/rh.EC_average; cd - 


 mris_spherical_average -erode 1 -orig white -t 0.4 -o OAS30001_MR_d0757 label rh.entorhinal rh sphere.reg rh.EC_average rh.entorhinal_exvivo.label 


 mris_anatomical_stats -mgz -f ../stats/rh.entorhinal_exvivo.stats -b -l ./rh.entorhinal_exvivo.label OAS30001_MR_d0757 rh white 

#--------------------------------------------
#@# Hippocampal Subfields processing Thu Jan 01 00:00:00 CST 1970

 mkdir -p /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults

 kvlSegmentHippocampalSubfields.sh OAS30001_MR_d0757 left /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults 


 kvlSegmentHippocampalSubfields.sh OAS30001_MR_d0757 right /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults 


 cp /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults/OAS30001_MR_d0757/left/segmentationWithoutPartialVolumingLog/posterior_left* /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri 


 cp /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults/OAS30001_MR_d0757/left/segmentationWithoutPartialVolumingLog/posterior_Left* /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri 


 cp /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults/OAS30001_MR_d0757/right/segmentationWithoutPartialVolumingLog/posterior_right* /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri 


 cp /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults/OAS30001_MR_d0757/right/segmentationWithoutPartialVolumingLog/posterior_Right* /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/mri 


 rm -rf /data/CNDA/build/OASIS/19700101_000000/fsrfer/OAS30001_MR_d0757/PROCESSED/OAS30001_MR_d0757/tmp/subfieldResults 

