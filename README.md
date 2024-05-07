# Demonstration Kernel on Uplift Modeling with Class Transformation Model and Oversampling

These kernels serve to provide sample demonstrations on uplift modeling using artificial datasets, particularly utilizing Class Transformation models and oversampling techniques.

## Brief Introduction to Uplift Modeling

### Basic Notations

With $i$ as index for $N$ individuals, the total effect $\tau_i$ is denoted as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/1.png?raw=true" width="180"/>
</p>

where, <br>
- $Y_i(1)$: outcome for person $i$ when he/she receives the active treatment
- $Y_i(0)$: outcome for person $i$ when he/she receives the control treatment
- $\tau_i$: total  effect of the treatment for that individual
<br>

Thus,
- if $\tau > 0$: treatment has a net positive effect on the individual $i$ and,
- if $\tau < 0$: negative effect
- if $\tau = 0$: no effect

Here, the Conditional Average Treatment Effect (CATE) is denoted as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/2.png?raw=true" width="350"/>
</p>

where $\tau(X_i)$ is the conditional average treatment effect for a subgroup of individuals characterized by the features $X_i$ which may include variables like age, gender, etc.
<br>
- $E[Y_i(1) | X_i]$ is the expected outcome for individuals in the subgroup $X_i$ when they receive the treatment.
- $E[Y_i(0) | X_i]$ is the expected outcome for individuals in the subgroup $X_i$ when they do not receive the treatment.
<br>

Thus if $CATE > 0$: on average, the treatment has a positive effect on the subgroup $X_i$.

As it is impossible to observe $Y_i(1)$ and $Y_i(0)$ at the same time, the *observed* $Y_i$ can be defined using $W_i$ as a binary variable having either 1 or 0 depending on if the individual $i$ receives the active (1) vs. control (0) treatment, as follows.

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/3.png?raw=true" width="350"/>
</p>

### Two-Model Approach
With basic notations defined as above, the simplest case of uplift modeling involves the Two-Model Approach, where the process can be summarized as:

1. Separate the Data: Divide the dataset into two subsets; one for those who received the active treatment (treatment group) and another for those who did not (control group).

2. Train two models
    - Train model $M1$ on the treatment group to predict $E[Y_i(1)|X_i]$, the expected amount spent given that the customer received the campaign.
    - Train another model $M2$ on the control group to predict $E[Y_i(0)|X_i]$, the expected amount spent given that the customer did not receive the campaign.
3. Use ML algorithm: You can use any machine learning algorithm like Random Forest or XGBoost for each model. These models will learn to predict the expected outcome based on the features $X_i$
4. Calculate Uplift: For each customer, use both models to predict the expected outcomes under treatment and control, and then calculate the uplift as $E[Y_i(1)|X_i] - E[Y_i(0)|X_i]$


Example outcome:
for a customer with features $X_i$ = {age 30, income 50,000}, let's say:
<br>
- $M1$ predicts $E[Y_i(1)|X_i]$ = \$250 <br>
- $M2$ predicts $E[Y_i(0)|X_i]$ = \$200 <br>

Then the Uplift would be \$250 - \$200 = \$50 <br>
Interpretation:
The uplift of $50 suggests that the marketing campaign is expected to increase the customer's spending by \$50.

Note that $X_i$ here are pointing to the different people (groups). <br>
- $M1$ is trained on $X_i$ features of people who actually received the treatment, and <br> 
- $M2$ is trained on the $X_i$ features of people who were in the control group

For instance if you have 200 people, you could divide them into two groups of 100 each: one that receives the treatment (e.g., exposed to a marketing campaign) and one that serves as the control group (not exposed to the campaign). Then,
1. Train Model $M1$: Use the 100 people in the treatment group to train a model M1 that predicts the expected amount they will spend( $E[Y_i(1)|X_i]$ ) based on their features $X_i$
2. Train model $M2$: use the 100 people in the control group to train $M2$ and $E[Y_i(0)|X_i]$
3. Calculate CATE:
For a new individual with features $X_i$, you would use both $M1$ and $M2$  to predict their expected spending under both scenarios (treatment and control). The difference between these two amounts would be the Conditional Average Treatment Effect (CATE) for that individual, calculated as $E[Y_i(1)|X_i] - E[Y_i(0)|X_i]$

### Class Transformation Model
Cass Transformation model, on the other hand, utilizes a newly created target variable, defined as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/4.png?raw=true" width="400"/>
</p>

where, <br>
- $Y_i^{observed}$: 1 if made a purchase, 0 otherwise <br>
- $W_i$: 1 if exposed to the (marketing) campaign, 0 otherwise <br>

Therefore, $Z_i$ can be essentially defined as:
- If received campaign and reacted (purchased), then $Z_i$ = 1 (Successful Treat)
- If not received campaign and not reacted (not purchased), then $Z_i$ = 1 (Successful Control)
- If received campaign and not reacted, then $Z_i$ = 0 (Unsuccessful Treat)
- If not received campaign, but still reacted, then $Z_i$ = 0 (Unsuccessful Control)

Now, if the following assumption holds:
<br>
*(which, in other words, connotes that for any given set of features X, the probability of those features being in the treatment group (campaigned) should be equal to the probability of those features being in the control group (not campaigned).
<br>
Note that this is why the number of treated customers should be equal to the number of control customers.
<br>
having an actual 1:1 ratio of treatment to control samples would be one way to satisfy this assumption, especially if you're randomly assigning individuals to each group. If the groups are perfectly balanced, then the probability of any given feature profile X being in either group would NATURALLY be 1/2.)*

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/5.png?raw=true" width="200"/>
</p>

Then $\tau(X_i)$ can be denoted as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/6.png?raw=true" width="250"/>
</p>

where: <br>
- $P(Z_i=1|X_i)$: conditional probability of $Z_i$ being 1 given the features $X_i$ <br>
- $2P(Z_i=1|X_i)-1$: scaling to make $\tau(X_i)$ range from -1 to 1 <br>
- $\tau(X_i)$ depicts CATE, i.e. the effect of the treatment such as marketing promotion. <br>
    - Thus if $CATE$ is close to 1, the effect is positive (treatment has positive effect on the purchase decision) <br>
    - If close to -1, the effect is negative (treatment has negative effect on the individuals' purchase decision; i.e., if they recieve the treatment, it's likely that they will not purchase) <br>
    - if close to 0, no effect on the purchase decision regardless of the campaign.

### Evaluation
> Uplift Curve

As for the evaluations of the models performance comparison, the Uplift Curve is defined as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/8.png?raw=true" width="300"/>
</p>

where, <br>
1. $Y_t^T$ and $Y_t^C$ refer to the outcomes (visible outcome), i.e. the actual observed purchase count, aligned from the largest CATE to the lowest CATE (t=1 being largest and so on).
2. $N_t$ refers to the number of observations.
3. The action of subtracting (-) as $Y^T$ - $Y^C$ is to subtract the response (purchase amount) regardless of the treatment effect from the response generated from the treatment.
This substraction gives you a measure of the "net effect" of the treatment on the observed outcomes.
4. The action of using $N^T$ and $N^C$ (multiplying or dividing them) is for balancing (i.e. to cope with potential data imbalance).

The procedure is as follows.
1. get uplift score (i.e. $CATE$, or, $\tau(X)$ ) from the model.
2. sorting: You then sort all individuals in your dataset by this predicted uplift score, from highest to lowest.
3. Top $t$ Observations: You take only the first $t$ individuals from this sorted list. These are the people for whom your model predicts the largest positive effect from the treatment.
4. Calculating $f(t)$: Finally, you use only these top $t$ observations to calculate the function $f(t)$ as defined in the equation.

Essentially, we are calculating $f(t)$ by using the subgroup of individuals who respond most positively to the campaign, and thus making the curve of $f(t)$.

An example Uplift Curve may look like this:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/uplift_curve.png?raw=true" width="550"/>
</p>

- The random line (orage): A positive slope connotes that in general treating the whole population has a positive effect.
- The points along the AUC line (blue): Connotes the predicted uplift gain
- Shape of the curve: If bell-shaped, implies the strong existence of both positive and negative effects in the dataset. We can say that no such effects exist if the curve is closer to the random line.

> Qini Curve

In addition to the uplift curve, Qini curve (or, Qini coefficient, which refers to the area under the Qini curve) can be used as a complementary evaluation metric, with $g(t)$ defined as:

<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/9.png?raw=true" width="220"/>
</p>

An example Qini curve may look like:
<p align="center">
  <img src="https://github.com/DSsoli/uplift_CT/blob/main/imgs/qini_curve.png?raw=true" width="550"/>
</p>

Note that, <br>

1. Qini coefficient is defined as the area under the Qini curve, and thus this is a single integer/float (i.e. of the area size).
2. AUUC is defiend as the area under the uplift curve, and this is also a single integer/float (the size of the area).
3. In balanced cases, both curves are essentially similar.
    - $g(t)$, i.e. Qini can be transformed into $f(t)$ via scaling. So the only difference between the two is how they are scaled (normalized)
5. Generally speaking, the larger the Qini or AUUC, the better the model is at identifying who will respond positively to the treatment
6. For imbalanced cases, Qini curve is considered to be more robust.