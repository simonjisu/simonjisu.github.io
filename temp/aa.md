$$\underset{a}{\max}\ q_\pi(s, a) \geq \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$$

holds because the maximum of a set of numbers is always greater than or equal to the average of those numbers. Here, the set of numbers is the Q-values for all possible actions in state s, i.e., {qπ(s, a) : a ∈ A(s)}.

Let's suppose that the maximum Q-value in state s is qπ(s, a*), where a* is the optimal action. Then we have:

$qπ(s, a*) ≥ qπ(s, a)$ for all a ∈ A(s)

Summing both sides of this inequality over all actions in A(s) and dividing by |A(s)|, we get:

$\dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a*) \geq \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$

Using the fact that the average of a set of numbers is equal to the sum of those numbers divided by the number of elements, we can rewrite this as:

$qπ(s, a*) ≥ \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$

Therefore, we have shown that the maximum Q-value qπ(s, a*) is greater than or equal to the average Q-value for all actions in state s, which gives us the desired inequality:

$\underset{a}{\max}\ q_\pi(s, a) \geq \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$


---


$$π'(a \vert s) = \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} + (1 - \epsilon) \pi(a \vert s) \text{ if } a = \arg\max_{b \in \mathcal{A}(s)} q_\pi(s, b) \\
π'(a \vert s) = \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \text{ otherwise}$$

$$q_\pi(s, \pi'(s)) = \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + (1 - \epsilon) \cdot \underset{a}{\max}\ q_\pi(s, a)$$

$$\underset{a}{\max}\ q_\pi(s, a) \geq \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$$

$$q_\pi(s, \pi'(s)) \geq \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + (1 - \epsilon) \sum_a \dfrac{\pi(a \vert s) - \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert}}{1 - \epsilon} q_\pi(s, a)$$

---


Starting from the expression:

$q_\pi(s, \pi'(s)) = \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + (1 - \epsilon) \cdot \underset{a}{\max}\ q_\pi(s, a)$

We want to replace the maximum term with an average over all actions. To do so, we use the fact that the maximum is greater than or equal to the average:

$\underset{a}{\max}\ q_\pi(s, a) \geq \dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$

Multiplying both sides of this inequality by (1 - ε), we get:

$(1 - \epsilon)\underset{a}{\max}\ q_\pi(s, a) \geq (1 - \epsilon)\dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$

Expanding the right-hand side of the inequality, we get:

$(1 - \epsilon)\dfrac{1}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) = \dfrac{1 - \epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a)$

Substituting this inequality into the expression for qπ(s, π'(s)), we get:

$q_\pi(s, \pi'(s)) \geq \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + \dfrac{1 - \epsilon}{\vert \mathcal{A}(s)\vert} \sum_a \underset{\substack{\text{action } a' \ \text{other than } a}}{\sum} \pi(a' \vert s) q_\pi(s, a)$

Simplifying the second term by distributing the summation over π(a' | s), we get:

$\dfrac{1 - \epsilon}{\vert \mathcal{A}(s)\vert} \sum_a \underset{\substack{\text{action } a' \ \text{other than } a}}{\sum} \pi(a' \vert s) q_\pi(s, a) = \dfrac{1 - \epsilon}{\vert \mathcal{A}(s)\vert} \sum_a \sum_{a' \neq a} \pi(a' \vert s) q_\pi(s, a)$

We can simplify this further by recognizing that the sum over all actions a' other than a is equal to the sum over all actions, minus the term for action a:

$\sum_{a' \neq a} \pi(a' \vert s) = \sum_a \pi(a \vert s) - \pi(a \vert s) = 1 - \pi(a \vert s)$

$q_\pi(s, \pi'(s)) \geq \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + \dfrac{1 - \epsilon}{\vert \mathcal{A}(s)\vert} \sum_a \dfrac{\pi(a \vert s) - \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert}}{1 - \epsilon} q_\pi(s, a)$