type t = {
  dropper_st : int;
  dropper_ttd : int;
  checker_st : int;
  checker_ttd : int;
}

let initial =
  {
    dropper_st = 0;
    dropper_ttd = 0;
    checker_st = 0;
    checker_ttd = 0;
  }
