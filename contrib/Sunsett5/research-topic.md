**1. Aggregate productivity growth**

**Introduction**
  
Aggregate productivity constantly grows due to technological innovation. This results in a long-run growth of real output. According to Real business-cycle theory, the sudden change in production technology also leads to fluctuation of real output from its long-run trend. This idea is, however, not well supported empirically.
  
I plan to investigate this idea under the assumption that sudden change in technology leads to excessive speculation and "irrational exuberant," a phrase used by Alan Greenspan during the 1990s dot-com bubble. This line of idea shares some similarities with the Austrian theory of business cycle but differs in that the cause of output gap is due to market speculation and not monetary authorities.

**Methodology**

I have had some experience with structural models such as Bayesian VAR. There is a MATLAB toolbox created by the ECB specifically for dealing with this kind of model. For structural models other than VAR, a software such as Time Series Lab (TSL) developed by Andrew Harvey could be useful. The data used will consist of common aggregate statistics such as real GDP, price level, interest rate, proxies for technological innovation, and data from stock markets to account for short term irrational expectations.

**Expectation**

I expect the result to be a comparison of this business model with other more traditional models. I might be able to determine how much each model contributes to the actual figures observed (assuming that the reality is a mix of the two). 

<br/>
   
**2. Changes in labor share and capital share of private income**

**Introduction**

There are 2 inputs in the simple production function: Labor and Capital. In return, the two factors receive income as a share of total private income. Lately, there has been a decline in the labor share of private income. IMF attributed half of the decline to technologial advancement, namely information and telecommunication technology and automation.[1](https://www.imf.org/en/Blogs/Articles/2017/04/12/drivers-of-declining-labor-share-of-income) In other words, the demand for labor and the proportion of labor-intensive industry is decreasing.

The simplification of production function such as Cobb-Douglas function is useful for studying aggregate changes; however, technological impact on skilled (high-income) and low-skilled (low-skilled) is different. Thus, analyzing them separately could yield useful insights.

**Methodology**

I will model the production function by separateing the labor portion into skilled and low-skilled labor. The different impacts of technological innovation (as measured by aggregate productivity) will be investigated. Furthermore, the effect of increasing capital stock will also be examined.

**Expectations**

The effect of technological might be or might not be long term for the two labor groups. If the structural shift is long term, then the breakpoint is expected to coincide with significant historical technological advancement. If the shift is temporary, Markov-switching model might be employed instead. This will tie back to the 1st topic of aggregate productivity which we hypothesized to be transitional and bounded to return to the equilibrium.
