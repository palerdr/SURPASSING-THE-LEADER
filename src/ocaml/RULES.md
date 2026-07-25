# Drop The Handkerchief -- Game Rules

Adapted from the "Surpassing The Leader" arc of the manga *Usogui* by Toshio Sako.

## Executable parity profile

The engine uses literal action seconds 1 through 60. A successful check is
inclusive:

$$
\mathrm{ST}=c-d+1 \qquad (c\ge d).
$$

During the leap window only Baku while acting as Dropper may choose second 61.
Checker remains capped at 60, and both players know this structural rule from
game initialization.

The executable revival model is identity-neutral. Let \(s\) be ST already in
the vial before a failed-check injection, \(t\) be the player's prior accrued
TTD, and \(q=s+60\) be the injected duration. For eligible states,

$$
\begin{aligned}
P_{\mathrm{rev}}(s,t)
&=0.80\left(1-\frac{s}{240}\right) \\
&\quad\times 2^{-\left(t/120\right)^{1.3}} \\
&\quad\times \max\!\left(0.4,\;0.88^{t/60}\right),
\end{aligned}
$$

A current dose \(q\ge300\) is fatal; \(t+q>300\) is fatal; and \(t+q=300\)
remains eligible when \(q<300\). CPR count and character physicality remain
narrative facts but are not probability inputs.

## The Game

Drop The Handkerchief is a Kakerou Match -> a formal high-stakes gamble. Two players take turns as the **Dropper (D)** and the **Checker (C)**. The Dropper holds a handkerchief while the Checker faces away. Within a 60-second turn, D must drop the handkerchief exactly once and C must turn around (check) exactly once. Neither player knows when the other will act. Roles swap each half-round.

- If C checks **at or after** the drop: the check **succeeds**. C is safe but
  accumulates inclusive **Squandered Time (ST)**, `check - drop + 1`.
- If C checks **before** the drop: the check **fails**, and C is punished.

## The Near-Death Drug (NDD)

Each player has a **cylinder/vial** that fills with near-death drug over the course of the game.

- **On a successful check:** The Squandered Time is added to C's cylinder as NDD. If the cylinder reaches its max capacity of **5 minutes**, it is injected immediately.
- **On a failed check:** A **1-minute penalty** is added to the cylinder, then the entire cylinder is injected.

When injected, the player's heart stops for the duration stored in the cylinder. The referee presiding over the game then attempts to revive them with CPR. If revival fails, that player **dies and loses**. If revived, the cylinder resets to zero and the match continues as soon as possible. The NDD begins to dissolve in the body after 1-2 hours, so the game is played with urgency.

The longer someone has been dead cumulatively, the harder revival becomes.
Referee fatigue is represented only through the TTD-derived factor in the
two-variable executable model. Death is a resource to be managed, not simply
avoided.

## The Leap Second

The game clock starts at **8:00 AM**. At exactly **8:59:60 AM**, a real-world leap second is inserted, giving that turn **61 seconds** instead of 60. A player using the "safe strategy" -> checking at the very end of the turn to guarantee a successful check -> will be caught off guard, because the Dropper can drop on the 61st second after the Checker has already looked.

The leap second is Baku's hidden weapon. Since deaths and their associated downtime shift the game clock forward, the timing of which player is D or C during the leap second turn can be manipulated. Baku's strategy revolves around engineering deaths to ensure the leap second falls on a turn where he is the Dropper -> this is called the **Leap Second Route (LSR)**.

- **LSR active:** The leap second turn coincides with Baku as D (he can exploit it).
- **LSR inactive:** The leap second turn coincides with Baku as C (useless to him).
- There are 4 distinct LSR routes depending on the roles and the timing between turns

## The Characters (Not as important)

### Baku (Madarame Baku / "Usogui")
The protagonist, a legendary gambler known as "The Lie Eater." Baku's strength is reading people -> he deconstructs opponents' strategies through observation, misdirection, and psychological manipulation. He has poor physicality compared to Hal, making each death far more dangerous for him. Baku is the one who knows about the leap second and built his entire strategy around exploiting it, going as far as manipulating Yakou into using a speaking clock (instead of a physical one) during the pre-game setup to ensure accurate timekeeping.

### Hal (Kiruma Souichi / "The Leader")
The antagonist and a genius-level strategist. Hal possesses extraordinary perception -> he can read micro-expressions and minute body-language to detect the handkerchief on the ground, and psychologically dismantle opponents through fear inducement and bluffing. He has superior physicality, making him far more likely to survive deaths. Hal is **not** consciously aware of the leap second, but his subconscious recognizes it, influencing his decisions in ways he doesn't fully understand. Hal's strategy centers on "trading deaths" -> deliberately dying to shift the game clock and manipulate the match state, while relying on his physical resilience to survive.

### The Referee (Yakou Hikoichi)
Yakou won the right to preside over Surpassing The Leader through a competition among referees. He administers injections, performs CPR, and enforces the rules. His effectiveness at revival degrades with each successive attempt. Yakou has a deep personal connection to Hal, having known him since childhood, and is burdened by the knowledge that if Hal loses, he will be the one to take his life.

## Violations (Auto-Loss)

1. Committing any violent acts or damaging the devices.
2. Not "dropping" or "checking" within 1 minute.
3. Any action that deliberately stalls the game.
4. Not following the referee's instructions.
