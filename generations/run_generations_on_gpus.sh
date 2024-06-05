# #!/bin/bash

#region GEMMA-7B BASELINE
# =============================================================================
# GEMMA-7B BASELINE
# =============================================================================

# NO
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32' > ./output/7b/baseline/7b_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 2' > ./output/7b/baseline/7b_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 1' > ./output/7b/baseline/7b_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 8' > ./output/7b/baseline/7b_wd.out 2>&1 &
#endregion
#region BASELINE INSTRUCTION TUNED









# #region BASELINE INSTRUCTION TUNED
# # =============================================================================
# # BASELINE INSTRUCTION TUNED
# # =============================================================================

# # NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
# --file test_no_context_not-packed.jsonl \
# --batch_size 32' 
# #> ./output/it/it_no.out 2>&1 &

# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_rel_context_not-packed.jsonl \
# --batch_size 4' > ./output/it/it_rel.out 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_random_context_not-packed.jsonl \
# --batch_size 2' > ./output/it/it_rand.out 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_wd_context_not-packed.jsonl \
# --batch_size 4' > ./output/it/it_wd.out 2>&1 &
# #endregion
# #region BASELINE INSTRUCTION TUNED
# # =============================================================================
# # BASELINE NON INSTRUCTION TUNED
# # =============================================================================

# # NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
# --file test_no_context_not-packed.jsonl \
# --batch_size 32' > ./output/nit/nit_no.out 2>&1 &

# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_rel_context_not-packed.jsonl \
# --batch_size 4' > ./output/nit/nit_rel.out 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
# --file test_random_context_not-packed.jsonl \
# --batch_size 2' > ./output/nit/nit_rand.out 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
# --file test_wd_context_not-packed.jsonl \
# --batch_size 4' > ./output/nit/nit_wd.out 2>&1 &

# #endregion











# #region IT TRAINED NO CONTEXT
# # =============================================================================
# # IT TRAINED NO CONTEXT
# # =============================================================================

# # NO

# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
# --file test_no_context_not-packed.jsonl \
# --batch_size 32 \
# --model mini_no_context_model.pt'  > ./output/trained_no/no_t_no.out 2>&1 &

# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_rel_context_not-packed.jsonl \
# --batch_size 4 \
# --model mini_no_context_model.pt' > ./output/trained_no/no_t_rel.out 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
# --file test_random_context_not-packed.jsonl \
# --batch_size 2 \
# --model mini_no_context_model.pt'  > ./output/trained_no/no_t_rand.out 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
# --file test_wd_context_not-packed.jsonl \
# --batch_size 4 \
# --model mini_no_context_model.pt'  > ./output/trained_no/no_t_wd.out 2>&1 &

# #endregion
# #region IT TRAINED REL CONTEXT
# # =============================================================================
# # IT TRAINED REL CONTEXT
# # =============================================================================

# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
# --file test_no_context_not-packed.jsonl \
# --batch_size 32 \
# --model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_no.out 2>&1 &

# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_rel_context_not-packed.jsonl \
# --batch_size 4 \
# --model mini_rel_context_model.pt' > ./output/trained_rel/rel_t_rel.out 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
# --file test_random_context_not-packed.jsonl \
# --batch_size 2 \
# --model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_rand.out 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
# --file test_wd_context_not-packed.jsonl \
# --batch_size 4 \
# --model mini_rel_context_model.pt'  > ./output/trained_rel/rel_t_wd.out 2>&1 &

# #endregion
# #region IT TRAINED RANDOM CONTEXT
# # =============================================================================
# # IT TRAINED RANDOM CONTEXT
# # =============================================================================
# # NO
# nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
# --file test_no_context_not-packed.jsonl \
# --batch_size 32 \
# --model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_no.out 2>&1 &

# # REL
# nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
# --file test_rel_context_not-packed.jsonl \
# --batch_size 4 \
# --model mini_random_context_model.pt' > ./output/trained_rand/rand_t_rel.out 2>&1 &

# # RANDOM
# nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
# --file test_random_context_not-packed.jsonl \
# --batch_size 2 \
# --model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_rand.out 2>&1 &

# # WD
# nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
# --file test_wd_context_not-packed.jsonl \
# --batch_size 4 \

# --model mini_random_context_model.pt'  > ./output/trained_rand/rand_t_wd.out 2>&1 &

# #endregion
# #region IT TRAINED WD CONTEXT
# =============================================================================
# IT TRAINED WD CONTEXT
# =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model mini_wd_context_model.pt' > ./output/trained_wd/wd_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model mini_wd_context_model.pt'  > ./output/trained_wd/wd_t_wd.out 2>&1 &


#endregion
#region NIT TRAINED NO
# =============================================================================
# NIT TRAINED NO
# =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_trained_no/nit_no_t_no.out 2>&1 &  

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_no_context_model.pt' > ./output/nit/nit_trained_no/nit_no_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_trained_no/nit_no_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_trained_no/nit_no_t_wd.out 2>&1 &

#endregion
#region NIT TRAINED REL
# =============================================================================
# NIT TRAINED REL
# =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model nit_mini_rel_context_model.pt'  > ./output/nit/nit_rel_trained/nit_trained_rel/nit_rel_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_rel_context_model.pt' > ./output/nit/nit_rel_trained/nit_trained_rel/nit_rel_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model nit_mini_rel_context_model.pt'  > ./output/nit/nit_rel_trained/nit_trained_rel/nit_rel_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_rel_context_model.pt'  > ./output/nit/nit_rel_trained/nit_trained_rel/nit_rel_t_wd.out 2>&1 &


#endregion
#region NIT TRAINED RAND
# # =============================================================================
# # NIT TRAINED RAND
# # =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model nit_mini_rand_context_model.pt'  > ./output/nit/nit_rand_trained/nit_trained_rand/nit_rand_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_rand_context_model.pt' > ./output/nit/nit_rand_trained/nit_trained_rand/nit_rand_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model nit_mini_rand_context_model.pt'  > ./output/nit/nit_rand_trained/nit_trained_rand/nit_rand_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_rand_context_model.pt'  > ./output/nit/nit_rand_trained/nit_trained_rand/nit_rand_t_wd.out 2>&1 &


#endregion
#region NIT TRAINED WD
# # =============================================================================
# # NIT TRAINED WD
# # =============================================================================
nohup sh -c 'CUDA_VISIBLE_DEVICES=0 python generate_predictions.py \
--file test_no_context_not-packed.jsonl \
--batch_size 32 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_no_trained/nit_trained_wd/nit_wd_t_no.out 2>&1 &

# REL
nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python generate_predictions.py \
--file test_rel_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_no_context_model.pt' > ./output/nit/nit_no_trained/nit_trained_wd/nit_wd_t_rel.out 2>&1 &

# RANDOM
nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python generate_predictions.py \
--file test_random_context_not-packed.jsonl \
--batch_size 2 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_no_trained/nit_trained_wd/nit_wd_t_rand.out 2>&1 &

# WD
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python generate_predictions.py \
--file test_wd_context_not-packed.jsonl \
--batch_size 4 \
--model nit_mini_no_context_model.pt'  > ./output/nit/nit_no_trained/nit_trained_wd/nit_wd_t_wd.out 2>&1 &
