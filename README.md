# BEP Planning

## Week 2

- Literature reading;
- project definition;
- validate ideas;
- project planning.

## Week 3

- Definieer functies voor metrics;
- Start data verzamelen huidige setup.
- Start introductie

## Week 4

- Implementeer no-pooling;
- LogDensity met AdvancedVI --> zit vast;
- verzamel data;
- vergelijk met week 3.

### Meeting notes 14 mei (review week 3)

- MLE 25 starting points, ADVI 1 (vaker runnen of in code implemteren)
- vergelijking pakt median individual obv error, vergelijk dezelfde individuen (opgelost door alles te plotten)
- Is de error hoger voor de uitersten van de physio metrics (plot corr) check
- posterior plotten check
- verschillende initializations ADVI (vaker runnen)
- define logdensity problem --> nog niet gelukt

## Week 5

- Varieer data door groep te oversamplen;
- kwantificeer effect;
- Start methode schrijven.

- Add heatmap
- Add eucledian distance, correlation dist. vs error and linear regression
- Add multiple starting points and use validation data (x)
- Take mean or take tip of distribution?

## Week 6 (26 mei)

Meeting inplannen met Shauna, oefen presentatie.
Max halverwege week op vakantie.

- Voorbereiding presentatie;
    a) cUDE
    b) VI
    c) Preliminary results (verschil methodes)
    d) outlook, wat ga ik nog doen
- Varieer prior (varantie van beta);
- Verzamel data;
- Vizualiseer data voor presentatie.
- Feedback verwerken introductie
- Methode afmaken

- mode estimation (kan ook via histogram) (x)
- betas train opnieuw fitten na vastzetten nn-params (x)
- prior door MLE op een kleine subset is mogelijk tracktable, alleen means daarop initializeren. Variance wat hoger
- methode cUDE staat onderaan.
- KL min, KL median, KL max?

## Week 7 (2 juni)

Max op vakantie. Meeting met Shauna.

- BEP midway presentation 4 juni;
- Feedback verzamelen en laatste weken herzien.

### BEP to-do-list June, 2th

#### Midterm presentation

- Less results in slides;
- red box where you want them to look;
- Rewrite cheat sheet:
  - clearly explain project goal;
  - drop extensive model explanation.

#### Coding

- train MLE with the same parameters as PP and NP;
- What if the dataset is smaller;
- Estimate priors on smaller subset.

#### Meetings

- Plan new BEP-meetings;
- Q2Max: Look at glucose values for extreme patients;
- Q2Max: Extreme data-split (maximum KL-divergence) or non-stratisfied split

#### Report

- Add training method to methods.

## Week 8 (9 juni)

- Verwerk feedback, verfijn;
- Schrijf resultaten.

## Week 9

- Draft inleveren;
- Discussie;
- Conclusie;
- Abstract.

## Week 10

- Feedback verwerken
- Hand-in en finalization. 24 juni 12:00
