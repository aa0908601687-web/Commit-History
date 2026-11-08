From Stdlib Require Import Arith Lia List.
Import ListNotations.

(* Abstract index-level specification for a fixed eight-candidate pool. *)
Definition valid_index (i : nat) : Prop := i < 8.
Definition valid_selection (xs : list nat) : Prop :=
  length xs = 6 /\ NoDup xs /\ Forall valid_index xs.

Theorem valid_selection_has_six :
  forall xs, valid_selection xs -> length xs = 6.
Proof. intros xs [H _]. exact H. Qed.

Theorem valid_selection_indices_bounded :
  forall xs i, valid_selection xs -> In i xs -> i < 8.
Proof.
  intros xs i [_ [_ H]] Hin.
  rewrite Forall_forall in H. exact (H i Hin).
Qed.

Theorem valid_selection_no_duplicates :
  forall xs, valid_selection xs -> NoDup xs.
Proof. intros xs [_ [H _]]. exact H. Qed.

(* Abstract boolean cardinality condition used by 8->6 QUBO integration.
   This is NOT a proof of Python floating-point energy or the CUDA-Q adapter. *)
Definition six_cardinality_penalty (k : nat) : nat :=
  if Nat.eqb k 6 then 0 else 1.

Theorem cardinality_penalty_zero_iff_six :
  forall k, six_cardinality_penalty k = 0 <-> k = 6.
Proof.
  intro k. unfold six_cardinality_penalty. split.
  - destruct (Nat.eqb k 6) eqn:Heq.
    + intro Hzero. apply Nat.eqb_eq in Heq. exact Heq.
    + intro Hfalse. discriminate Hfalse.
  - intro H. subst k. reflexivity.
Qed.

Theorem valid_selection_zero_penalty :
  forall xs, valid_selection xs ->
  six_cardinality_penalty (length xs) = 0.
Proof.
  intros xs H. unfold six_cardinality_penalty.
  rewrite (valid_selection_has_six xs H).
  reflexivity.
Qed.

(* Actual QUBO coefficient mapping, float arithmetic, solver bit endianness,
   and exact argmax remain runtime / differential validation obligations. *)

(* Index/bitvector representation invariant used by the Python 8-candidate
   mapping. This establishes the length of the encoded boolean vector only;
   it does not prove bit endianness or floating-point QUBO agreement. *)
Definition candidate_mask (xs : list nat) : list bool :=
  map (fun i => if in_dec Nat.eq_dec i xs then true else false) (seq 0 8).

Theorem candidate_mask_has_eight_bits :
  forall xs, length (candidate_mask xs) = 8.
Proof.
  intro xs. unfold candidate_mask.
  rewrite map_length. apply seq_length.
Qed.
