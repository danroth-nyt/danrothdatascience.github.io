---
layout: post
title: "Perceptual Evaluation of a Music Source Separation CNN Trained With Binaural and Ambisonic Audio"
author: "Dan Roth"
categories: research
tags: [research, music, source-separation, cnn]
image: "sourcesep.png"
---

### Abstract

This research explores the idea of using different spatial audio formats for training
music source separation neural networks.  DeepConvSep, a library designed by Marius
Miron, Pritish Chandna, Gerard Erruz, and Hector Martel, is used as a framework for testing
different convolutional neural networks for source separation.  A listening test is then
detailed and test results are analyzed in order to perform a perceptual evaluation of the
models.  Conclusions are drawn regarding the effectiveness of using spatial audio formats
for training source separation neural networks.

## Introduction

Neural networks for audio seek to enable an artificial intelligence to speak and hear
akin to a human.  There have been many obstacles in the development of this field as
audio contains an abundance of information, ranging from tens of thousands of different
frequency and phase components as well as external, unwanted noise and the potential for
multiple sources.  In order to solve this problem, several methods currently exist to
condense and reformat audio into data vectors that are more compatible with input into a
neural network.

The advent of spatial audio formats such as binaural and ambisonic audio has
created the potential for more “natural sounding” audio than stereo.  However, there have
been little prior studies into whether or not these more “natural sounding” audio formats
would benefit a neural network in its attempt to learn and understand sound.  This
research seeks to experiment with this idea by training and perceptually evaluating several
source separation neural network models that are trained with different audio formats.

## Source Separation

### Conceptual Foundation

Source separation is a process that aims to separate audio mixtures into their
respective source elements, whether it be music or speech, etc.  The fields of automatic
speech recognition (ASR), music post-production, and music information retrieval (MIR)
have all benefitted from research into improvements to source separation techniques [1].
Source separation can be further subdivided into two distinctive categories.  Audio Quality
Oriented (AQO) source separation seeks to separate the source signals from a mix while
best preserving their perceptual audio fidelity.  In contrast, Significance Oriented (SO)
approaches extract high-level, semantic information from separated source signals [1].
This research focuses on the AQO approach in assessing the effectiveness of source
separation.

### Research Inspiration

Source separation is an ideal task for a neural network as it is essentially a
classification task.  DeepConvSep, designed by Marius Miron, Pritish Chandna, Gerard
Erruz, and Hector Martel, is a convolutional neural network (CNN) library that was built for
the Signal Separation Evaluation Campaign (SiSEC) [2, 3].  The original neural network
model was trained with the Demixing Secrets DSD100 database in order to separate
contemporary/pop music.  The DSD100 is comprised of a set of 100 songs each with their
corresponding mixtures and stems separated into four categories (vocals, bass, drums, and
other) originally developed for SiSEC 2017.

Although the DSD100 model was originally trained using monaural audio input,
researcher Gerard Erruz explored the idea of training the model with multi-channel and
even spatial audio input [1].  Erruz originally hypothesized that binaural spectral cues such
as interaural level and time differences (ILD and ITD) could have a positive effect on
helping a source separation CNN perform more accurately.  In order to test this, the
DSD100 was binaurally processed in several configurations (“cross”, “xeix”, and “random”)
for input into the CNN.

> A listening test was conducted in order to perceptually
evaluate these spatially trained source separation neural
networks.

Five objective measures were used in order to quantify this performance
improvement: source to distortion ratio (SDR), source to inference ratio (SIR), source to
noise ratio (SNR), source to artifacts ratio (SAR), as well as the incorporation of an image
to spatial distortion ratio (ISR) to account for spatial distortion in stereo outputs [1].  Erruz
concluded that certain measures (SDR, SIR) indicated that models trained with binaural
audio input (excepting a configuration that placed all training signals in random positions)
outperformed models with stereo or monaural input [1].

While these objective measures appear to be a sensible way of demonstrating
differences in model performance, this source separation task still must appeal to the AQO
approach and present a perceptually high-fidelity audio output.  A listening test was
conducted in order to perceptually evaluate these spatially trained source separation neural
networks.  In addition, a model using the same CNN architecture was trained with an
ambisonically encoded DSD100 in order to explore the possibility of using alternative forms
of spatially processed audio to further improve source separation performance.

## Test Methodology

### Choice of Stimuli

Several CNN models had to be trained to represent the different approaches.  The
original DeepConvSep architecture was preserved so that the differences in database
inputs would be the primary variable.  One model was trained using the standard DSD100
as a stereo-trained model.  The binaurally and ambisonically trained models were both
configured in the “cross” formation as it had been noted that signal predominance in a
particular channel improved separation performance [1].  Figure 1 illustrates the spatial
positioning for the dataset.

![figure1](../assets/img/posts/sourcesep/fig1.png)

<p style="text-align: center;"><sub>Figure 1. “Cross” layout for binaural decoding.  “D”, “V”, “B”, and
“O” correspond to drums, vocals, bass, and other respectively. [1]</sub></p>

The DSD100 was binaurally processed using Two!Ears’ binaural scene simulator and
the SADIE head-related transfer function (HRTF) database in order to create a BDSD100
(binaural DSD100) [4].  An ADSD100 (ambisonic DSD100) was created by ambisonically
encoding the database using SN3D normalized spherical harmonics.  This experiment used
1st order ambisonics, yielding a 4-channel signal that was binaurally decoded.  The
binaural decoding was performed by convolving the ambisonic audio with SADIE HRTF’s
that were ambisonically encoded using Google’s resonance audio toolkit and summing the
resultant signals into left and right channels [5].

Models were trained using the various versions of the DSD100 and then the
database’s 50 test songs were separated using each model.  The test stimuli were chosen
from these 50 songs, all of which were not a part of the original model training.  Models
were all trained using a single mean-square error training stage and the default
parameters of the DeepConvSep CNN illustrated by table 1.

![table1](../assets/img/posts/sourcesep/table1.png)

<p style="text-align: center;"><sub> Table 1. CNN training parameters [1].</sub></p>

A Microsoft Azure cloud computer with a Tesla K80 graphics card and an Intel Xeon
E5-2690 V3 CPU was used for model training.

### Test Configuration

A listening test was conducted with a total of 14 participants, all of whom self-
identified as having normal hearing.  Most participants were experienced audio engineers
although there were a variety of audio experience levels amongst the group.  Tests were
conducted in the hemi-anechoic chamber at the University of Derby and participants took
the test using a pair of Beyerdynamic DT770 PRO (80 ohm) closed-back headphones.
These measures were taken to reduce the influence of any room reflections in an
acoustically untreated space, which could skew localization cues.  However, it may have
been of interest to consider other types of headphones.  For example, it has been noted
that free-air equivalent coupling (FEC) headphones have more accurate reproduction of
binaural stimuli as they do not alter the acoustic impedance of the ear canal when covering
the ears [6].

![figure2](../assets/img/posts/sourcesep/fig2.jpeg)

<p style="text-align: center;"><sub>Figure 2. Listening test setup in University of Derby’s hemi-
anechoic chamber.</sub></p>

Tests were administered on a computer using a DragonFly Red DAC.  A Multiple
Stimulus with Hidden Reference and Anchors (MUSHRA) test was designed using the Web
Audio Evaluation Tool [7].  The Web Audio Evaluation Tool was set to calibrate the test
signals to -23 LUFS and participants did not have the ability to change signal levels.  A
MUSHRA test format was chosen in order to conform to the ITU’s method for the
subjective assessment of intermediate quality level of coding systems (ITU-R BS.1534-1)
[8].  The MUSHRA format has been a predominant paradigm in several source separation
listening tests and is noted for its ability to assess the degradation of a test signal as
compared to a known reference [9].

The test consisted of a total of 8 songs and participants were asked to either rate
how well separated chosen elements were (such as bass, drums, etc.) or how “stable”
different audio samples were.  Participants were told that audio was more stable if it did
not feature unrealistic amplitude and pan modulation and was unstable if the inverse was
true.  Rating was done on a scale from 0 to 100.  A hidden reference was created by
taking test songs’ respective stems from the DSD100 database.  A low-range hidden
anchor was created by applying a low-pass filter to the hidden reference at 3.5 kHz.  The
remaining options were comprised of separations performed by the ambisonic, binaural,
and stereo trained models.  The null hypothesis for the experiment was that ambisonic and
binaurally trained models would be perceptually no different than the original stereo-
trained model.  A sample page of the test can be seen in Figure 3.

![figure3](../assets/img/posts/sourcesep/fig3.png)

<p style="text-align: center;"><sub>Figure 3. Sample page of the MUSHRA listening test.</sub></p>

## Analysis of Results

One-way analysis of variance (ANOVA) tests were performed on separation and
stability response data in MATLAB in order to capture overall trends and to confirm
statistical significance.  P-scores for separation and stability responses were 0.3863 and
0.001 respectively.  This indicates that separation means did not prove to have statistically
significant differences as the p-score was above the 0.05 (5%) threshold to prove
significance.  The different formats may be too indistinguishable in terms of separation
quality and none of them may have fulfilled the intended task.  It may have also been
difficult to determine what separation quality meant for the participants, and this could be
an area for future improvement in related listening examinations.  However, the stability
rating fell below the p-score threshold and thus rejected the null hypothesis.  The observed
loss in stability when using binaural and ambisonic audio to train stereo separation models
was therefore confirmed, indicating that this may not be an ideal approach to use when
training source separation models as stability can have a large impact on perceptual
evaluation.  Notched box plots were also produced from this analysis (see Figures 4 and 5).

![figure4](../assets/img/posts/sourcesep/fig4.jpeg)

<p style="text-align: center;"><sub>Figure 4. Notched box plot comparing separation ratings between
differently trained CNN models.  “Ambi”, “anch”, “bina”, “ref”, and
“ster” refer to ambisonic, anchor, binaural, reference, and stereo.</sub></p>

![figure5](../assets/img/posts/sourcesep/fig5.jpeg)

<p style="text-align: center;"><sub>Figure 5. Notched box plot comparing stability ratings between
differently trained CNN models.</sub></p>

All of the source separation models scored poorly overall, with the stereo model
featuring a slightly higher median and lower variance of opinion.  However, as all of the
notches of the models’ box plots are nearly aligned this observation falls outside of the
confidence interval, denoting that there is not strong evidence to conclude that the
medians actually differ.  Ambisonic and binaural models each had greater variance and
more outliers overall, which could mean that these formats produced perceptually
confusing results that yielded a variety of opinions.  There are many outliers for the
reference and these could very well represent participants that were not fully clear about
the listening task; these outliers could also be consistent throughout.

Stability scores had greater variance, but also drew more definite conclusions once
again.  The stereo model confidently outperformed the binaural and ambisonic models as
its median score was higher and outside of the confidence intervals of the other models.
Strangely enough, ratings for the stereo model were greatly varying.  Identifying audio
stability may be too ambiguous and could potentially use some refinement as a target
metric in the future.  For the most part it does appear that models trained with ambisonic
and binaural audio featured less stability when separating stereo music; this could likely be
attributed to the differences in spatial positioning techniques between these formats.

There were no major conclusions to be drawn from any specific genre or instrument
from the test selection as the models responded very differently to the various types of
music and instruments.  However, it was observed that the stereo model scored
consistently better in terms of vocal separation than the other two models.  Perhaps vocal
separation was easier to distinguish in contrast to other instruments, thus participants were
able to make more decisive choices.

## Conclusions

Models trained with ambisonic and binaural audio did not outperform the stereo
trained model.  It is possible that a model trained with stereo material simply performs
best when separating stereo audio.  It would be of interest to further investigate a variety
of different audio formats separated by their respectively trained models to observe any
performance improvements when pairing model training with intended target material.
Higher order ambisonics could also be used in order to observe any differences when using
higher resolution ambisonic input.

This experiment also throws a shadow of doubt over the objective measurements
used in the original research [1].  There may have to be better measures available when
quantifying source separation system performance in order to assure that the quality of
differing source separation models can be accurately and thoroughly understood.

Overall, it is apparent this source separation algorithm was not perceptually effective.
While specific combinations of model and input material yielded excellent separation
results, the performance was inconsistent and further strides will have to be made to
improve source separation neural network architectures.  The results of this study indicate
that altering the spatial format of audio training material may not be the path to progress
for source separation at this point.

## References

[1] G. Erruz, “Binaural Source Separation with Convolutional Neural Networks,” 2017.
Masters Thesis.  Universitat Pompeu Fabra. https://zenodo.org/record/1095835#.XNLm2-
hKiUk (accessed: 28 March, 2019)

[2] P. Chandna, M. Miron, J. Janer, and E. Gomez, “Monoaural audio source separation
using deep convolutional neural networks,” in International Conference on Latent Variable
Analysis and Signal Separation, 2017.

[3] A. Liutkus, F. R. Stöter, Z. Rafii, D. Kitamura, B. Rivet, et al., “The 2016 Signal
Separation Evaluation Campaign,” in 13th International Conference on Latent Variable
Analysis and Signal Separation (LVA/ICA 2017), Grenoble, France, 2017, pp. 323 - 332,
10.1007/978-3-319-53547-0_31. hal-01472932

[4] “Two!Ears,” http://twoears.eu/ (accessed: 8 April, 2019).

[5] Google, “Resonance Audio,” https://resonance-audio.github.io/resonance-audio/
(accessed: 8 April, 2019).

[6] T. McKenzie, D. Murphy, and G. Kearney, “Diffuse-field Equalisation of First-order
Ambisonics,” in 20th International Conference on Digital Audio Effects (DAFx-17),
Edinburgh, UK, 5–9 September, 2017, pp. 8.

[7] N. Jillings, D. Moffat, B. De Man, and J. D. Reiss, “Web Audio Evaluation Tool: A
browser-based listening test environment,” in 12th Sound and Music Computing
Conference, 2015.

[8] E. Cano, D. FitzGerald, and K. Brandenburg, “Evaluation of quality of sound source
separation algorithms: Human perception vs quantitative metrics,” in 2016 24th European
Signal Processing Conference (EUSIPCO), 2016, pp. 1758–1762.

[9] H. Wierstorf, D. Ward, R. Mason, E. M. Grais, C. Hummersone, and M. D. Plumbley,
“Perceptual Evaluation of Source Separation for Remixing Music,” in Audio Engineering
Society Convention 143, 2017.
