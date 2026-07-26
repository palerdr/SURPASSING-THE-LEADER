type solution = {
  value : float;
  dropper_strategy : float array;
  checker_strategy : float array;
}

val solve : float array array -> solution
