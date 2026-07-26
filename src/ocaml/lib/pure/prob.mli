type outcome =
  | Terminal of float
  | Live of State.t

type branch = {
  probability : float;
  outcome : outcome;
}

val revival_probability : int -> int -> float
val transition : State.t -> int -> int -> branch list
