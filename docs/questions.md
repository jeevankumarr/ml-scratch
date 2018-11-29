
### Question for Practice
__SQL:__
1. Calculate monthly active users
2. Calculate monthly chur rate
3. Calculate returning user rate
4. Calcualte ARPDAU
5. Calculate request acceptance rate BY date, month
6. Calculate request abandon rate BY date, month

__Unsorted:__  
7. How do you proof that males are on average taller than females by knowing just gender or height?
8. What is a monkey patch
9. How to AB test non-normal variables?
10. Create a network graph from a file


__Stats:__
1. What is hypotheses testing?
2. What is Type I and Type II Error?
3. How to calculate sample size?
4. What corrections are made to correct for multiple testing? Pros, Cons. 
5. How to test proportions? Normal and non-normal data.
6. How to test means? Normal and non-normal data.
7. How to test counts? Normal and non-normal data.



__Probability:__
1. Bobo the amoeba has 25%, 25%, and 50% change of producing 0, 1, 2 offspring, respectively. Each of Bobo's descendants also have the same probabilities. What is the probability that Bobo's lineage dies out?
    ```
    p = 0.25 + 0.25 * p + 0.5 * p^2
    ```

2. In any 15 min interval there is 20% probability that you will see at least one shooting star. What is the prob that you will see at least one shooting star in the period of an hour?
    ```
    p(no shooting star in 15 min) = 1 - 0.20 = 0.80
    p(no shooting star in 60 min) = 0.8 ^ 4
    p(at leat 1 shooting star in 60 min) = 1 - 0.8^4
    ```
3. How can you generate random no. between 1 - 7 with only one die?
    ```
    Throw the die twice, and assign number as (throw_1 + throw_2) % 7
    except when it is 6, 6
    ```

4. How can you get a fair coin toss if the coin is weighted to heads more often than tails?
    ```
    H, T = T
    T, H = H
    
    H, H = ignore
    T, T = ignore
    ``` 

5. You have a 50-50 mixture of two normal distribution with the same sd. How far apart do the mean needs to be in order for this distribution to be bi-modal?
    ```text
    a distribution is bi modal when <a href="https://www.codecogs.com/eqnedit.php?latex=|&space;\mu_1&space;-&space;\mu_2&space;|&space;>&space;2\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|&space;\mu_1&space;-&space;\mu_2&space;|&space;>&space;2\sigma" title="| \mu_1 - \mu_2 | > 2\sigma" /></a>
    ```    

6. Given draws from a normal distribution with known parameters, how can you simulate draws from a uniform distribution?

7. A certain couple tells you that they have two children, at least one of which is a girl. What is the probability that they have two girls?

8. You have a group of couples that decide to have children until they have the first girl, after which they stop having children. What is the expected gender ratio of the children each couple will have?

9. How many can you split 12 people into 3 teams of 4?

10. Your hash function assigns each object to a number between 1:10, each with equal probability of a hask collision? What is the expected number of hash collisions? What is the expected number of hashes that are unused. 

11. You call 2 Ubers and 3 Lyfts. If the time that each takes to reach you is IID, what is the probability that all the Lyfts arrive first? What is the probability all Ubers arrives first?
    ```
    probability that 1 Lyft is arrives first = 3/5
    probability that 2 Lyft is arrives first (after the first Lyft) = 2/4
    probability that 3 Lyft is arrives first (after the first Lyft) = 1/3
    prob of all Lyfts arriving before the Ubers = 3/5 * 2/4* 1/3 = 1/10
    prob of all Ubers arriving before the Ubers = 1/5 * 2/4
    ```

12. Fizz Buzz
    ```python
    def fizz_buzz(n): 
        for i in range(n+1):
            if i%3 == 0 and i%5 == 0: print('FizzBuzz')
            elif i%3 == 0: print('FizzBuzz')
            elif i%5 == 0: print('Fizz')
            else print(i)
    ```

13. On a dating site, users can select 5 out of 24 adjectives to describe themselves. A match is declared between two users if they match on at least 4 adjectives. If Alice and Bob randomly pick adjectives, what is the probability that they form a match?
    ```text
        No. of ways to pick 5 adjectives = 25C5 = No. of ways Alice picks adjectives.
        No. of ways for Bob to pick matching 4 matching adjectives = 5C4 * 19C1
        No. of ways for Bob to pick matching 5 matching adjectives = 5C5 * 19C0
        p(match) = 5C4 * 19C1 + 5C5 * 19C0 / 24C5 = 0.002 
    ```

14. A lazy high school senior types up application and envelopes to n different colleges, but puts the applications randomly into the envelopes. What is the expected number of application that went to the right college?

15. Let's say that you have a very tall father. On average, what would you expect the hieght of his son to be? Taller, equal, or shorter? What if you had a very short father?

16. What is the expected number of coin flips until you get two heads in a row? What is the expected no. of coin flips until you get two tails in a row?

17. Let's say we play a game where i keep flipping a coin until I get heads. If the first time i get heads is on the nth coin, then i pay you 2n-1 dollars. How much would you pay me to play this game?

18. You have two coins, one of which is fair and comes up heads with probability of 1/2, and the other which is biased and comes up heads with prob of 3/4. You randomly pick coin and flip it twice, and get heads both times. What is the prob that you picked the fair coin?
    ```
    Goal: p(fair | 2 heads)
        = p(2 heads | fair) * p(fair) / p(2 heads)
	*  p(fair) = p(unfair) = 0.5
	*  p(2 heads | fair) = 2C2 0.5^2 * 0.5^0 = 0.25
	*  p(2 heads) = p(2 heads | fair) * p(fair) + p(2 heads | unfair) * p(unfair) 
	*  p(2 heads | unfair) =  2C2 0.75^2 * 0.25^0 = 0.5625
	*  p(fair | 2 heads) = 0.25 * 0.5 / (0.5625 * 0.5 + 0.25 * 0.5)
	*  p(fair | 2 heads) = 0.3076
    ```
    
19. You have 0.1% chance of picking up a coin with both heads, and a 99.9% chance that you pick up a fair coin. You flip your coin and it comes up heads 10 times. What's the chance that you picked the fair coin, given the information that you observed?
    ```
	 * Goal: p(fair | 10h ) = p(10h | fair) * p(fair) / p(10h)
	 * p(fair) = 0.999, p(unfair) = 1-p(fair)
	 * p_10_heads_given_fair = sc_misc.comb(10, 10) * np.power(0.5, 10) * np.power(0.5, 0) = 0.0009766
	 * p_10_heads = p_10_heads_given_unfair * p_unfair + p_10_heads_given_fair * p_fair  = 0.001976
	 * p_fair_giv_10h = 0.4938
    ```

20. What is the probability of Full House, Flush, Royal Flush, ... 
21. What is the expected no. of heads of a coin that increases p(head) by 0.1 each time it is tossed. Starting p(head) = 0.1.

__Problem Solving:__
1. How to identify fake news?
2. How to identify fraud profiles?
3. How to detect bots?
4. How to estimate birthday?
5. How to measure newsfeed health?
6. How to map nick names to real names?


__Computer Vision and Deep Learning:__
1. What is NN?
2. What is CNN?
3. What is RNN?

__Programming:__
1. How do you get the count of each letter in a sentence
2. Distinct first names from a list?
3. How to measure user engagement
4. Write a function to calculate all possible assignment vectors of 2n users, where n users in control and n in treatment?
5. Given a list of tweets, determine the top 10 most used hashtags. 
6. Knapsack problem
7. Travelling Salesman Problem
8. Reservoir Sampling
9. Calculate the square root of number
10. Given a list of numbers return outliers
11. When can parallelism make your algo run faster? When could it make your algorithms run slower?










