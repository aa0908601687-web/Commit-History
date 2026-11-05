From Rocq Require Import Arith Lia List.
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
  apply Forall_forall in H. exact (H i Hin).
Qed.

Theorem valid_selection_no_duplicates :
  forall xs, valid_selection xs -> NoDup xs.
Proof. intros xs [_ [H _]]. exact H. Qed.

(* These proofs establish abstract selection invariants only.
   The Python/CUDA-Q adapters and exact-search argmax still require verification. *)
