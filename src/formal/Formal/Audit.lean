import Lean
import Formal.MatrixGame.Basic

/-!
# Axiom audit

`#audit_axioms` checks every theorem and definition whose name starts with
`Formal.`. It fails the build when a declaration depends on an axiom other
than `propext`, `Classical.choice`, and `Quot.sound`. This rejects `sorry`
(`sorryAx`), compiler-trusted evaluation (`Lean.ofReduceBool` from
`native_decide`), and any user-declared axiom.

`FormalAudit.lean` at the project root imports the whole library and runs the
command, so `lake build` performs the audit.
-/

open Lean Elab Command

namespace Formal.Audit

/-- The axioms of Lean's standard foundation. -/
def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- Fail when a `Formal.*` declaration depends on a nonstandard axiom, and
report how many declarations were checked. -/
elab "#audit_axioms" : command => do
  let env ← getEnv
  let mut checked : Nat := 0
  let mut bad : Array (Name × Array Name) := #[]
  for (name, info) in env.constants.toList do
    unless (`Formal).isPrefixOf name do continue
    if name.isInternal then continue
    match info with
    | .thmInfo _ | .defnInfo _ | .opaqueInfo _ =>
      checked := checked + 1
      let axioms ← liftCoreM (collectAxioms name)
      let extra := axioms.filter (fun a => !allowedAxioms.contains a)
      unless extra.isEmpty do
        bad := bad.push (name, extra)
    | .axiomInfo _ =>
      bad := bad.push (name, #[name])
    | _ => pure ()
  if bad.isEmpty then
    logInfo m!"axiom audit: {checked} declarations use only propext, Classical.choice, Quot.sound"
  else
    let lines := bad.toList.map fun (n, axs) => m!"{n}: {axs.toList}"
    throwError m!"axiom audit failed:\n{MessageData.joinSep lines "\n"}"

end Formal.Audit
