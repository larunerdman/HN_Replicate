

###
### ORIGINAL MODEL
###


library(readxl)
library(reshape2)
library(rjson)
library(RJSONIO)
library(V8)
library(ggplot2)

### FUNCTIONS

get_hn_only = function(in_df){
  left_df = in_df[in_df$Side == 1 & in_df$view_side == "Left",]
  right_df = in_df[in_df$Side == 2 & in_df$view_side == "Right",]
  both_df = in_df[in_df$Side == 3,]
  
  out_df = rbind(left_df, right_df, both_df)
  
  return(out_df)
}

get_sen_spec = function(score_vec, target, threshold){
  score0 = score_vec[target == 0]
  score1 = score_vec[target == 1]
  
  # browser()
  
  tnr = sum(score0 < threshold)/length(score0)
  fnr = sum(score1 < threshold)/length(score1)
  
  out_list = list("specificity" = list("est" = tnr),
                  "sensitivity" = list("est" = 1-fnr))
  return(out_list)
}

get_sen_spec_withCI = function(score_vec, target, threshold, n_it, out_vecs=FALSE){
  
  out_list = get_sen_spec(score_vec=score_vec,
                          target=target,
                          threshold=threshold)
  n = length(score_vec)
  
  sen_vec = c()
  spec_vec = c()
  for(i in 1:n_it){
    set.seed(i)
    
    # browser()
    samp = sample(1:n,size = n,replace = TRUE)
    true_tnr_fnr = get_sen_spec(score_vec=score_vec[samp],
                                target=target[samp],
                                threshold=threshold)
    sen_vec = c(sen_vec, true_tnr_fnr$sensitivity)
    spec_vec = c(spec_vec, true_tnr_fnr$specificity)
  }
  
  # browser()
  sen_vec = sort(unlist(sen_vec))
  spec_vec = sort(unlist(spec_vec))
  
  out_list[["sensitivity"]][["ci"]] = quantile(sen_vec,c(0.025, 0.975))
  out_list[["specificity"]][["ci"]] = quantile(spec_vec,c(0.025, 0.975))
  
  if(out_vecs){
    
    out_list[["sensitivity"]][["vec"]] = c(sen_vec, rep(NA, n_it - length(sen_vec)))
    out_list[["specificity"]][["vec"]] = c(spec_vec, rep(NA, n_it - length(spec_vec)))
    
    
  }
  
  return(out_list)
  
}  

sen_spec_split = function(score_vec, target, split_vec, threshold, n_it, out_vecs=FALSE){
  split_vals = unique(split_vec)
  
  out_list = list()
  
  for(split_val in split_vals){
    
    # browser()
    
    split_score = na.omit(score_vec[split_vec == split_val])
    split_target = na.omit(target[split_vec == split_val])
    
    out_list[[split_val]] = get_sen_spec_withCI(score_vec = split_score, 
                                                target = split_target, 
                                                threshold = threshold, 
                                                n_it = n_it, 
                                                out_vecs = out_vecs)
  }
  
  return(out_list)  
}

add_platt_preds = function(in_list, scale_ref = "val", epoch = 1){
  in_list_names = names(in_list)
  to_scale = in_list_names[in_list_names != scale_ref]
  
  folds = length(in_list[[scale_ref]])
  
  ## creating and applying scaling model
  for(fold in 1:folds){
    
    threshold = get_fnrs(in_list = in_list, epoch = epoch,fold = fold, return_threshold=TRUE)
    
    # browser()
    
    scale_set = data.frame(label = in_list[[scale_ref]][[fold]][[epoch]][["target"]], 
                           pred = in_list[[scale_ref]][[fold]][[epoch]][["pred"]])
    
    scale_mod = glm(label ~ pred, scale_set, family = binomial)    
    
    in_list[[scale_ref]][[fold]][[epoch]][["pred_scaled"]] = predict(scale_mod, newdata = data.frame(pred = in_list[[scale_ref]][[fold]][[epoch]][["pred"]]),type = "response")
    in_list[[scale_ref]][[fold]][[epoch]][["threshold"]] = predict(scale_mod, newdata = data.frame(pred = threshold), type = "response")
    
    for(group in to_scale){
      in_list[[group]][[fold]][[epoch]][["pred_scaled"]] = predict(scale_mod, newdata = data.frame(pred = in_list[[group]][[fold]][[epoch]][["pred"]]),type = "response")
    }
  }  
  
  # browser()
  
  return(in_list)
}

get_most_recent = function(in_dataset){
  
  out_dataset = data.frame(matrix(nrow = 0, ncol = ncol(in_dataset)))
  names(out_dataset) = names(in_dataset)
  
  for(id in unique(in_dataset$pt_id)){
    pt_df = in_dataset[in_dataset$pt_id == id, ]
    max_row = pt_df[pt_df$USNumber == max(as.numeric(pt_df$USNumber)),][1,]
    
    out_dataset = rbind(out_dataset,max_row)
  }
  
  names(out_dataset) = names(in_dataset) 
  
  return(out_dataset)
}

make_st_df = function(st_df, st_details){
  
  st_details$STID = paste0("STID", st_details$ID, "_", st_details$view_side, "_", st_details$US_num, "_1")
  
  st_merged = merge(st_df, st_details, by.x = "id", by.y = "STID")
  head(st_merged)
  st_merged$pt_id = unlist(lapply(strsplit(st_merged$id,"_"),function(x){x[1]}))
  length(unique(st_merged$pt_id))
  
  st_uniq = st_merged[!duplicated(st_merged$ID),]
  table(st_uniq$target)
  
  ## confirming HN only 
  st_hn_only = get_hn_only(in_df = st_merged)
  
  st_hn_only$id_num = as.numeric(substr(st_hn_only$id,5,7))
  
  st_hn_only$sex = factor(st_hn_only$Sex, levels = c(1,2), labels = c("male","female"))
  st_hn_only$target01 = ifelse(st_hn_only$target == "no Obstruction", 0, 1)
  st_hn_only$apd_groups = factor(st_hn_only$apd_groups, levels = c("under_6", "apd_6_to_9", "apd_9_to_14", "over_14", "unmeasured"))
  st_hn_only$apd_groups[is.na(st_hn_only$apd_groups)] = "unmeasured"
  st_hn_only$HN_Side_fac = factor(st_hn_only$Side, levels = 1:3, labels = c("Left", "Right", "Both"))
  st_hn_only$USNumber = factor(st_hn_only$US_num, levels = sort(unique(st_hn_only$US_num)))
  
  st_hn_only$age_grps = NA
  st_hn_only$age_grps[st_hn_only$age_at_US_wk < 102] = "under2"
  st_hn_only$age_grps[st_hn_only$age_at_US_wk >= 102 & 
                        st_hn_only$age_at_US_wk < 260] = "age2to5"
  st_hn_only$age_grps[st_hn_only$age_at_US_wk >= 260] = "over5"
  st_hn_only$age_grps_fac = factor(st_hn_only$age_grps, levels = c("under2","age2to5","over5"))
  
  st_hn_only$pcode = toupper(substr(st_hn_only$Postal.Code,1,1))
  st_hn_only$PostalCode = factor(st_hn_only$pcode, levels = c("M", "N", "L", "P", "K"))
  
  st_hn_only$prep = ifelse(st_hn_only$id_num <= 300, "prep1", "prep2")
  st_hn_only$prep = factor(st_hn_only$prep, levels = c("prep1","prep2"))
  
  st_hn_only$US_machine[is.na(st_hn_only$US_machine)] = "Unknown"
  
  ### most recent only dataset: 
  hn_only_most_recent = get_most_recent(in_dataset = st_hn_only)
  
  out_list = list("overall" = st_hn_only,"most_recent" = hn_only_most_recent)
  
  return(out_list)
}

make_stan_df = function(stan_df, stan_datasheet, stan_train){
  stan_df$target01 = ifelse(stan_df$target == "no Obstruction", 0, 1)
  
  head(stan_datasheet)
  head(stan_df)
  stan_datasheet$img_id = paste0(stan_datasheet$anon_mrn,"_",stan_datasheet$`side of hydronephrosis`,"_",stan_datasheet$anon_accession,"_1")
  
  #### GET MACHINE INFORMATION !! 
  
  stan_merge = merge(stan_df, stan_datasheet, by.x = "id", by.y = "img_id")
  head(stan_merge)
  table(table(stan_merge$anon_mrn))
  table(stan_merge$target, stan_merge$sex)
  table(stan_merge$sex)
  table(stan_merge$`side of hydronephrosis`,stan_merge$target)
  
  stan_uniq = stan_merge[!duplicated(stan_merge$anon_mrn),]
  table(stan_uniq$target)
  table(tolower(stan_uniq$ethnicity))
  
  dim(stan_merge)
  length(unique(stan_merge$anon_mrn))
  dim(stan_df)
  
  stan_merge$age_grps = NA
  stan_merge$age_grps[stan_merge$age < 2] = "under2"
  stan_merge$age_grps[stan_merge$age >= 2 & stan_merge$age <= 5] = "age2to5"
  stan_merge$age_grps[stan_merge$age > 5] = "over5"
  
  table(stan_merge$age_grps)
  table(stan_merge$age_grps, stan_merge$target)
  
  stan40_test = stan_merge[!(stan_merge$id %in% stan_train),]
  
  
  out_list = list("overall" = stan_merge, "test40" = stan40_test)
  
  return(out_list)
}

make_uiowa_df = function(ui_df, ui_datasheet, ui_train){
  ui_df$pt_id = unlist(lapply(strsplit(ui_df$id,"_"),function(x){x[1]}))
  ui_df$target01 = ifelse(ui_df$target == "no Obstruction", 0, 1)
  
  ui_merge = merge(ui_df, ui_datasheet, by.y = "Name", by.x = "pt_id")
  
  ui_merge$age_days = difftime(ui_merge$Ultrasound.Date, ui_merge$DOB, units = "days")
  
  ui_merge$age_grp = NA
  ui_merge$age_grp[ui_merge$age_days < 365*2] = "under2"
  ui_merge$age_grp[ui_merge$age_days >= 365*2 & ui_merge$age_days <= 365*5] = "age2to5"
  ui_merge$age_grp[ui_merge$age_days > 365*5] = "over5"
  
  ui40_test = ui_merge[!(ui_merge$pt_id %in% ui_train),]
  
  out_list = list("overall" = ui_merge, "test40" = ui40_test)
  
  return(out_list)
}


###
###
###       ORIGINAL MODEL
###
###


results_wd = "C:/Users/lauren erdman/OneDrive - SickKids/HNReplicateData_20211205/ModelResults/"

###
###     SILENT TRIAL DATA 
###


SENSITIVITY = 0.90

## No test data -- model usedto test Stanford and Silent trial
raw_in = readLines(paste0(results_wd,"SickKids_origST_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covTrue_OSFalse_20210613_v3.json"))
test = v8()
test$assign("dat", JS(raw_in))
st_stan_test = test$get("dat")
st_stan_test = add_platt_preds(in_list = st_stan_test, epoch = 10,sensitivity = SENSITIVITY)
st_stan_test = add_platt_preds(in_list = st_stan_test, epoch = 13,sensitivity = SENSITIVITY)
st_stan_test = add_platt_preds(in_list = st_stan_test, epoch = 16,sensitivity = SENSITIVITY)
st_stan_test = add_platt_preds(in_list = st_stan_test, epoch = 30,sensitivity = SENSITIVITY)

identical(st_stan_test$st[[1]][[30]]$id,
          st_stan_test$st[[1]][[33]]$id)

st_preds = data.frame(st_pred_mod_ep10 = st_stan_test$st[[1]][[10]]$pred_scaled,
                      st_pred_mod_ep13 = st_stan_test$st[[1]][[13]]$pred_scaled,
                      st_pred_mod_ep16 = st_stan_test$st[[1]][[16]]$pred_scaled,
                      st_pred_mod_ep30 = st_stan_test$st[[1]][[30]]$pred_scaled,
                      id = st_stan_test$st[[1]][[30]]$id,
                      target = st_stan_test$st[[1]][[30]]$target)
st_preds$target = factor(st_preds$target, levels = c(0,1), labels = c("no Obstruction", "Obstruction"))
head(st_preds)

fold=1
(thresh_ep30 = st_stan_test[["val"]][[fold]][[30]][["threshold"]])
ggplot(st_preds, aes(x = target, y = st_pred_mod_ep30, fill = target)) + geom_violin() + 
  geom_hline(yintercept = thresh_ep30) + ylim(0,1) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ylab("Prediction")

st_orig_list = make_st_df(st_df = st_preds,
                          st_details = read.csv(paste0(results_wd, "SilentTrial_Datasheet_deID_20211205.csv"),header = TRUE, as.is = TRUE))

st_hn_only = st_orig_list[["overall"]]
hn_only_most_recent = st_orig_list[["most_recent"]]


####
#### SILENT TRIAL ALL VISITS
####


st_overall_origmod = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep30, 
                                         target = st_hn_only$target01, 
                                         threshold = thresh_ep30, 
                                         n_it = 500, out_vecs = FALSE)

####
#### SILENT TRIAL MOST-RECENT ONLY 
####

st_mostrec_origmod = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep30, 
                                         target = hn_only_most_recent$target01, 
                                         threshold = thresh_ep30, 
                                         n_it = 500, out_vecs = FALSE)


###
###     STANFORD DATA 
###


stan_orig_list = make_stan_df(stan_df = read.csv(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/orig_st_results/stan_mod0_mod1_20210613_v3.csv"),
                              stan_datasheet = read_excel("C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/AI Hydronephrosis Data deidentified.xlsx"),
                              stan_train = readRDS("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/Stan60%train_ids.rds"))


stan_merge = stan_orig_list[["overall"]]
head(stan_merge)
stan40_test = stan_orig_list[["test40"]]


stan_all_origmod = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep30, 
                                       target = stan_merge$target01, 
                                       threshold = thresh_ep30, 
                                       n_it = 500)


stan_40test_origmod = get_sen_spec_withCI(score_vec = stan40_test$stan_pred_mod_ep30, 
                                          target = stan40_test$target01, 
                                          threshold = thresh_ep30, 
                                          n_it = 500)



###
###     UIOWA DATA 
###


ui_orig_list = make_uiowa_df(ui_df = read.csv(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/orig_st_results/ui_mod0_mod1_20210613_v3.csv"),
                             ui_datasheet = read.csv("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UIowa_Datasheet2.csv", header = TRUE, as.is = TRUE),
                             ui_train = readRDS("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/UI60%train_ids.rds"))


ui_df = ui_orig_list[["overall"]]
ui40_test = ui_orig_list[["test40"]]

ui_all_origmod = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep30, 
                                     target = ui_df$target01, 
                                     threshold = thresh_ep30, 
                                     n_it = 500)

## 40% test 
ui_40test_origmod = get_sen_spec_withCI(score_vec = ui40_test$ui_pred_mod_ep30, 
                                        target = ui40_test$target01, 
                                        threshold = thresh_ep30, 
                                        n_it = 500)




###
###
###       STANFORD FINE-TUNE MODEL
###
###
dat_thresh_ep30 = 0.03094032

###
###     SILENT TRIAL DATA 
###

st_hn_only = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ST_postStanfinetune_20210818.rds")

hn_only_most_recent = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/STmostrecent_postStanfinetune_20210818.rds")

head(st_hn_only)

st_overall_stft = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep30, 
                                      target = st_hn_only$target01, 
                                      threshold = dat_thresh_ep30, 
                                      n_it = 500, out_vecs = FALSE)

## most recent only 
st_mostrec_stft = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep30, 
                                      target = hn_only_most_recent$target01, 
                                      threshold = dat_thresh_ep30, 
                                      n_it = 500, out_vecs = FALSE)

###
###     STANFORD DATA 
###

stan_merge = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/Stanford_postStanfinetune_20210818.rds")

stan_overall_stft = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep30, 
                                        target = stan_merge$target01, 
                                        threshold = dat_thresh_ep30, 
                                        n_it = 500)

###
###     UIOWA DATA 
###
ui_df = readRDS("C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UI_postStanfinetune_20210818.rds")

ui_overall_stft = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep30, 
                                      target = ui_df$target01, 
                                      threshold = dat_thresh_ep30, 
                                      n_it = 500)



###
###
###       UI FINE-TUNE MODEL
###
###

dat_thresh_ep1 = 1

###
###     SILENT TRIAL DATA 
###
st_hn_only = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ST_postUIowafinetune_20210818.rds")
hn_only_most_recent = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/STmostrecent_postUIowafinetune_20210818.rds")


## overall 
st_overall_uift = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep1, 
                                      target = st_hn_only$target01, 
                                      threshold = dat_thresh_ep1, 
                                      n_it = 500)

## most recent only 
st_mostrec_uift = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep1, 
                                      target = hn_only_most_recent$target01, 
                                      threshold = dat_thresh_ep1, 
                                      n_it = 500, out_vecs = FALSE)


###
###     STANFORD DATA 
###
stan_merge = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/Stanford_postUIowafinetune_20210818.rds")

stan_overall_uift = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep1, 
                                        target = stan_merge$target01, 
                                        threshold = dat_thresh_ep1, 
                                        n_it = 500)

###
###     UIOWA DATA 
###
ui_df = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UI_postUIowatune_20210818.rds")


## overall 
ui_overall_uift = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep1,
                                      target = ui_df$target01,
                                      threshold = dat_thresh_ep1,
                                      n_it = 500)



###
###
###       STANFORD-SICKKIDS TRAINED MODEL
###
###


dat_thresh_ep20 = 0.06890174 

###
###     SILENT TRIAL DATA 
###

st_hn_only = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ST_CombinedTrain_20210818.rds")
hn_only_most_recent = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/STmostrecent_CombinedTrain_20210818.rds")

## overall
st_overall_skstcomb = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep20, 
                                          target = st_hn_only$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500, out_vecs = FALSE)

## most recent only 
st_mostrec_skstcomb = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep20, 
                                          target = hn_only_most_recent$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500, out_vecs = FALSE)


###
###     STANFORD DATA 
###
stan_merge = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/Stanford_STSKCombinedTrain_20210818.rds")

stan_overall_skstcomb = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep20, 
                                            target = stan_merge$target01, 
                                            threshold = dat_thresh_ep20, 
                                            n_it = 500)
###
###     UIOWA DATA 
###
ui_df = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UI_STSKCombinedTrain_20210818.rds")

ui_overall_skstcomb = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep20, 
                                          target = ui_df$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500)



###
###
###       UIOWA-SICKKIDS TRAINED MODEL
###
###

dat_thresh_ep20 = 0.0765223

###
###     SILENT TRIAL DATA 
###

st_hn_only = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ST_STUICombinedTrain_20210818.rds")

hn_only_most_recent = readRDS(hn_only_most_recent, file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/STmostrecent_STUICombinedTrain_20210818.rds")

## overall
st_overall_skuicomb = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep20, 
                                          target = st_hn_only$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500, out_vecs = FALSE)

## most recent only 
st_mostrec_skuicomb = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep20, 
                                          target = hn_only_most_recent$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500, out_vecs = FALSE)
###
###     STANFORD DATA 
###
stan_merge = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/Stanford_STUICombinedTrain_20210818.rds")


stan_overall_skuicomb = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep20, 
                                            target = stan_merge$target01, 
                                            threshold = dat_thresh_ep20, 
                                            n_it = 500)

###
###     UIOWA DATA 
###
ui_df = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UI_STUICombinedTrain_20210818.rds")

ui_overall_skuicomb = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep20, 
                                          target = ui_df$target01, 
                                          threshold = dat_thresh_ep20, 
                                          n_it = 500)



###
###
###       COMBINED TRAINED MODEL
###
###

dat_thresh_ep20 = 0.05574685

###
###     SILENT TRIAL DATA 
###
st_hn_only = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ST_CombinedTrain_20210818.rds")
hn_only_most_recent = readRDS(file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/STmostrecent_CombinedTrain_20210818.rds")
# 
# 
## overall
st_overall_allcomb = get_sen_spec_withCI(score_vec = st_hn_only$st_pred_mod_ep20, 
                                         target = st_hn_only$target01, 
                                         threshold = dat_thresh_ep20, 
                                         n_it = 500, out_vecs = FALSE)

## most recent only 
st_mostrec_allcomb = get_sen_spec_withCI(score_vec = hn_only_most_recent$st_pred_mod_ep20, 
                                         target = hn_only_most_recent$target01, 
                                         threshold = dat_thresh_ep20, 
                                         n_it = 500, out_vecs = FALSE)



###
###     STANFORD DATA 
###
stan_merge = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/Stanford/Stanford_CombinedTrain_20210818.rds")

stan_overall_allcomb = get_sen_spec_withCI(score_vec = stan_merge$stan_pred_mod_ep20, 
                                           target = stan_merge$target01, 
                                           threshold = dat_thresh_ep20, 
                                           n_it = 500)


###
###     UIOWA DATA 
###
ui_df = readRDS(file = "C:/Users/lauren erdman/OneDrive - SickKids/HN/UIowa/UI_CombinedTrain_20210818.rds")


ui_overall_allcomb = get_sen_spec_withCI(score_vec = ui_df$ui_pred_mod_ep20, 
                                         target = ui_df$target01, 
                                         threshold = dat_thresh_ep20, 
                                         n_it = 500)



###
#*******#
###
##
##      OVERALL PLOT
##
###
#*******#
###

names_vec = c("spec.est","spec2.5ci","sen.est","sen.2.5ci","spec.est1","spec97.5ci","sen.est1","sen97.5ci", "Model", "Dataset")

results_df = data.frame(rbind(c(data.frame(st_overall_origmod)[1,], data.frame(st_overall_origmod)[2,], "SickKids Trained", "SickKids Test"),
                              c(data.frame(st_mostrec_origmod)[1,], data.frame(st_mostrec_origmod)[2,], "SickKids Trained", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_all_origmod)[1,], data.frame(stan_all_origmod)[2,], "SickKids Trained", "Stanford Test"),
                              c(data.frame(ui_all_origmod)[1,], data.frame(ui_all_origmod)[2,], "SickKids Trained", "UIowa Test"),
                              
                              c(data.frame(st_overall_stft)[1,], data.frame(st_overall_stft)[2,], "Stanford fine-tuned", "SickKids Test"),
                              c(data.frame(st_mostrec_stft)[1,], data.frame(st_mostrec_stft)[2,], "Stanford fine-tuned", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_overall_stft)[1,], data.frame(stan_overall_stft)[2,], "Stanford fine-tuned", "Stanford Test"),
                              c(data.frame(ui_overall_stft)[1,], data.frame(ui_overall_stft)[2,], "Stanford fine-tuned", "UIowa Test"),
                              
                              c(data.frame(st_overall_uift)[1,], data.frame(st_overall_uift)[2,], "UIowa fine-tuned", "SickKids Test"),
                              c(data.frame(st_mostrec_uift)[1,], data.frame(st_mostrec_uift)[2,], "UIowa fine-tuned", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_overall_uift)[1,], data.frame(stan_overall_uift)[2,], "UIowa fine-tuned", "Stanford Test"),
                              c(data.frame(ui_overall_uift)[1,], data.frame(ui_overall_uift)[2,], "UIowa fine-tuned", "UIowa Test"),     
                              
                              c(data.frame(st_overall_skstcomb)[1,], data.frame(st_overall_skstcomb)[2,], "SickKids-Stanford Trained", "SickKids Test"),
                              c(data.frame(st_mostrec_skstcomb)[1,], data.frame(st_mostrec_skstcomb)[2,], "SickKids-Stanford Trained", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_overall_skstcomb)[1,], data.frame(stan_overall_skstcomb)[2,], "SickKids-Stanford Trained", "Stanford Test"),
                              c(data.frame(ui_overall_skstcomb)[1,], data.frame(ui_overall_skstcomb)[2,], "SickKids-Stanford Trained", "UIowa Test"),
                              
                              c(data.frame(st_overall_skuicomb)[1,], data.frame(st_overall_skuicomb)[2,], "SickKids-UIowa Trained", "SickKids Test"),
                              c(data.frame(st_mostrec_skuicomb)[1,], data.frame(st_mostrec_skuicomb)[2,], "SickKids-UIowa Trained", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_overall_skuicomb)[1,], data.frame(ui_overall_allcomb)[2,], "SickKids-UIowa Trained", "Stanford Test"),
                              c(data.frame(ui_overall_skuicomb)[1,], data.frame(ui_overall_skuicomb)[2,], "SickKids-UIowa Trained", "UIowa Test"),
                              
                              c(data.frame(st_overall_allcomb)[1,], data.frame(st_overall_allcomb)[2,], "SickKids-Stanford-UIowa\n Trained", "SickKids Test"),
                              c(data.frame(st_mostrec_allcomb)[1,], data.frame(st_mostrec_allcomb)[2,], "SickKids-Stanford-UIowa\n Trained", "SickKids Test\n Most Recent"),
                              c(data.frame(stan_overall_allcomb)[1,], data.frame(stan_overall_allcomb)[2,], "SickKids-Stanford-UIowa\n Trained", "Stanford Test"),
                              c(data.frame(ui_overall_allcomb)[1,], data.frame(ui_overall_allcomb)[2,], "SickKids-Stanford-UIowa\n Trained", "UIowa Test")
))


names(results_df) = names_vec
results_df = data.frame(lapply(results_df, unlist))
str(results_df)


head(results_df)
results_df$Model

graph_df = data.frame(Estimate = c(results_df$sen.est, results_df$spec.est),
                      Metric = c(rep("Sensitivity",nrow(results_df)),
                                 rep("Specificity",nrow(results_df))),
                      LCL = c(results_df$sen.2.5ci, results_df$spec2.5ci),
                      UCL = c(results_df$sen97.5ci, results_df$spec97.5ci),
                      Model = c(results_df$Model, results_df$Model),
                      Dataset = c(results_df$Dataset, results_df$Dataset))
head(graph_df)

graph_df$Model = factor(graph_df$Model, 
                        levels = c("SickKids Trained","Stanford fine-tuned",
                                   "UIowa fine-tuned","SickKids-Stanford Trained",
                                   "SickKids-UIowa Trained","SickKids-Stanford-UIowa\n Trained" ))

library(ggplot2)

theme_set(
  theme_bw(base_size = 15)
)

ggplot(graph_df, aes(x = Metric, y = Estimate, col = Metric)) + 
  geom_point(size = 2) + geom_errorbar(size = 1.2, mapping=aes(ymin = LCL, ymax = UCL)) + 
  scale_color_manual(values = c("Sensitivity" = "dodgerblue", "Specificity" = "deeppink2")) + 
  facet_grid(Dataset ~ Model)



