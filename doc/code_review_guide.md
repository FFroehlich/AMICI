# Code review guide

A guide for reviewing code and having your code reviewed by others.

## Everyone

* Don't be too protective of your code
* Accept that, to a large extent, coding decisions are a matter of personal
  preference
* Don't get personal
* Ask for clarification
* Avoid strong language
* Try to understand your counterpart's perspective
* Clarify how strong you feel about each discussion point

## Reviewing code

* If there are no objective advantages, don't force your style on others
* Ask questions instead of making demands
* Assume the author gave his best
* Mind the scope (many things are nice to have, but might be out of scope
  of the current change - open a new issue)
* The goal is "good enough", not "perfect"
* Be constructive
* You do not always have to request changes

## Having your code reviewed

* Don't take it personal - the review is on the code, not on you
* Code reviews take time, appreciate the reviewer's comments
* Assume the reviewer did his best (but might still be wrong)
* Keep code changes small (e.g. separate wide reformatting from actual code
  changes to facility review)
* If the reviewer does not understand your code, probably many others won't
  either

## Checklist

* [ ] Adherence to project-specific style guide
* [ ] The code is self-explanatory
* [ ] The code is concise / expressive
* [ ] Meaningful identifiers are used
* [ ] Corner-cases are covered, cases not covered fail loudly
* [ ] The code can be expected to scale well (enough)
* [ ] The code is well documented (e.g., input, operation, output), but
      without trivial comments
* [ ] The code is [SOLID](https://en.wikipedia.org/wiki/SOLID)
* [ ] New code is added in the most meaningful place (i.e. matches the
      current architecture)
* [ ] No magic numbers
* [ ] No hard-coded values that should be user inputs
* [ ] No dead code left
* [ ] The changes make sense
* [ ] The changes are not obviously degrading performance
* [ ] There is no duplicated code
* [ ] The API is convenient
* [ ] Code block length and complexity is adequate
* [ ] Spelling okay
* [ ] The code is tested
