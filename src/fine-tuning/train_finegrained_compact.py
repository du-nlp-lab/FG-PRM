import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb
import yaml
import nltk

from fgrlhf.ppo import PPOTrainer
from fgrlhf.policy import T5Policy
from fgrlhf.value import T5Value
from fgrlhf.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

from reward import FineGrainedCompactReward

logging.basicConfig(level=logging.ERROR)

# prepare accelerator and logger
accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)
        
#test_raw_data = {
#    'prompts_text': ['question: Who does angelina jolie play in kung fu panda? context: wikipage: Kung Fu Panda: Secrets of the Scroll text: Kung Fu Panda: Secrets of the Scroll Kung Fu Panda: Secrets of the Scroll is an animated short film in the "Kung Fu Panda" film series. It was included on the 2015 Digital HD and 2016 Blu-ray, and DVD re-release of "Kung Fu Panda" and "Kung Fu Panda 2". Jack Black, James Hong, Dustin Hoffman, Seth Rogen, David Cross, Randall Duk Kim and Lucy Liu reprise their roles from the movies, while Kari Wahlgren and James Sie replace Angelina Jolie and Jackie Chan as Tigress and Monkey, respectively. The introduction, which takes place a bit before "Kung Fu Panda 3", shows Po realizing that his dad, Mr Ping, has been giving away his things as bonus gifts to customers, including his toys of the Furious Five. As he searches for them, Oogway\'s narration asks whether there is any real difference between an accident and destiny, thus setting the scene for the story. 10 years before the events of the first "Kung Fu Panda", a younger Tigress is training under Shifu, but always seems to disappoint him by not conforming to his strict style. | wikipage: Kung Fu Panda (franchise) text: Kung Fu Panda (franchise) Kung Fu Panda is a media franchise by DreamWorks Animation, consisting of three films: "Kung Fu Panda" (2008), "Kung Fu Panda 2" (2011) and "Kung Fu Panda 3" (2016). The first two were distributed by Paramount Pictures, while the third film was distributed by 20th Century Fox. Three shorts, "Secrets of the Furious Five" (2008), "Kung Fu Panda Holiday Special" (2010) and "" (2011), were also released. A television series for Nickelodeon television network, "", premiered in 2011. A second series, "", was released on Amazon Prime in November 2018. The franchise, set in a fantasy wuxia genre version of ancient China populated by humanoid animals, features the adventures of Po Ping, a giant panda, who was improbably chosen as the prophesied Dragon Warrior. Although his status is initially doubted, Po proves himself worthy as he strives to fulfill his destiny and learn about his past with his new friends. The film series has been highly acclaimed with its first two features being nominated for the Academy Award for Best Animated Feature as well as numerous Annie Awards while the television series has won 11 Emmy Awards. | wikipage: Kung Fu Panda Holiday text: Kung Fu Panda Holiday Kung Fu Panda Holiday is a 2010 American computer-animated television special produced by DreamWorks Animation and directed by Tim Johnson. A spinoff of the "Kung Fu Panda" franchise, the special stars the voices of Jack Black, Angelina Jolie, Dustin Hoffman, Jackie Chan, Seth Rogen, David Cross, Lucy Liu, James Hong, and Jack McBrayer. The special premiered on NBC on November 24, 2010, and its premiere broadcast drew 5.9 million viewers. Master Shifu (Dustin Hoffman) assigns Po (Jack Black) to host the annual Winter Feast at the Jade Palace, a highly ritualized formal occasion where all the Kung Fu masters of China attend, insisting on the perfection of the event. Although excited, Po also wishes to spend the holiday with his father Mr. Ping (James Hong), and rejects the finest chefs in China in an attempt to have him cater the event. However, Mr. Ping believes Po is more concerned with his duties as the Dragon Warrior, and adamantly remains at his restaurant to feed the townsfolk who have nowhere else to eat for the holiday.', 'question: What was the period before the enlightenment called? context: wikipage: Age of Enlightenment text: Age of Enlightenment The Enlightenment (also known as the Age of Enlightenment or the Age of Reason) was an intellectual and philosophical movement that dominated the world of ideas in Europe during the 18th century, the "Century of Philosophy". French historians traditionally place the Enlightenment between 1715 (the year that Louis XIV died) and 1789 (the beginning of the French Revolution). International historians begin the period in the 1620s, with the start of the scientific revolution. "" () of the period widely circulated their ideas through meetings at scientific academies, Masonic lodges, literary salons, coffee houses and in printed books and pamphlets. The ideas of the Enlightenment undermined the authority of the monarchy and the Church and paved the way for the political revolutions of the 18th and 19th centuries. A variety of 19th-century movements, including liberalism and neo-classicism, trace their intellectual heritage to the Enlightenment. The Enlightenment included a range of ideas centered on reason as the primary source of authority and legitimacy and came to advance ideals like liberty, progress, tolerance, fraternity, constitutional government and separation of church and state. In France, the central doctrines of the Enlightenment philosophers were individual liberty and religious tolerance, in opposition to an absolute monarchy and the fixed dogmas of the Roman Catholic Church. | wikipage: Women in the Middle Ages text: Women in the Middle Ages Women in the Middle Ages occupied a number of different social roles. During the Middle Ages, a period of European history lasting from around the 5th century to the 15th century, women held the positions of wife, mother, peasant, artisan, and nun, as well as some important leadership roles, such as abbess or queen regnant. The very concept of "woman" changed in a number of ways during the Middle Ages and several forces influenced women\'s roles during their period. The Roman Catholic Church was a major unifying cultural influence of the Middle Ages with its selection from Latin learning, preservation of the art of writing, and a centralized administration through its network of bishops. Historically in the Catholic and other ancient churches, the role of bishop, like the priesthood, was restricted to men. The first Council of Orange (441) also forbade the ordination of deaconesses, a ruling that was repeated by the Council of Epaone (517) and the Second Council of Orléans (533). With the establishment of Christian monasticism, other roles within the Church became available to women. From the 5th century onward, Christian convents provided an alternative to the path of marriage and child-rearing, to play a more active religious role. | wikipage: Early modern period text: Early modern period The early modern period of modern history follows the late Middle Ages of the post-classical era. Although the chronological limits of the period are open to debate, the timeframe spans the period after the late portion of the post-classical age (c. 1500), known as the Middle Ages, through the beginning of the Age of Revolutions (c. 1800) and is variously demarcated by historians as beginning with the Fall of Constantinople in 1453, with the Renaissance period, and with the Age of Discovery (especially with the voyages of Christopher Columbus beginning in 1492, but also with Vasco da Gama\'s discovery of the sea route to the East in 1498), and ending around the French Revolution in 1789. Historians in recent decades have argued that from a worldwide standpoint, the most important feature of the early modern period was its globalizing character. The period witnessed the exploration and colonization of the Americas and the rise of sustained contacts between previously isolated parts of the globe. The historical powers became involved in global trade, as the exchange of goods, plants, animals, and food crops extended to the Old World and the New World. The Columbian Exchange greatly affected the human environment. New economies and institutions emerged, becoming more sophisticated and globally articulated over the course of the early modern period.', 'question: Who came up with the saying the customer is always right? context: wikipage: The customer is always right text: The customer is always right "The customer is always right" is a motto or slogan which exhorts service staff to give a high priority to customer satisfaction. It was popularised by pioneering and successful retailers such as Harry Gordon Selfridge, John Wanamaker and Marshall Field. They advocated that customer complaints should be treated seriously so that customers do not feel cheated or deceived. This attitude was novel and influential when misrepresentation was rife and "caveat emptor" (let the buyer beware) was a common legal maxim. Variations include ""le client n\'a jamais tort"" (the customer is never wrong) which was the slogan of hotelier César Ritz who said, "If a diner complains about a dish or the wine, immediately remove it and replace it, no questions asked". A variation frequently used in Germany is ""der Kunde ist König"" (the customer is king), while in Japan the motto ""okyakusama wa kamisama desu"" () meaning "the customer is a god", is common. However it was pointed out as early as 1914 that this view ignores that customers can be dishonest, have unrealistic expectations, and/or try to misuse a product in ways that void the guarantee. | wikipage: Harry Gordon Selfridge text: Harry Gordon Selfridge Harry Gordon Selfridge, Sr. (11 January 1858 – 8 May 1947) was an American-British retail magnate who founded the London-based department store Selfridges. His 20-year leadership of Selfridges led to his becoming one of the most respected and wealthy retail magnates in the United Kingdom. He was known as the \'Earl of Oxford Street\'. Born in Ripon, Wisconsin, Selfridge delivered newspapers and left school at 14 when he found work at a bank in Jackson, Michigan. After a series of jobs, Selfridge found a position at Marshall Field\'s in Chicago, where he stayed for the next 25 years. In 1890, he married Rose Buckingham, of the prominent Chicago Buckingham family. In 1906, following a trip to London, Selfridge invested £400,000 in his own department store in what was then the unfashionable western end of Oxford Street. The new store opened to the public on 15 March 1909, and Selfridge remained chairman until he retired in 1941. In 1947, he died of bronchial pneumonia at age 89. Selfridge was born to Robert Oliver Selfridge and Lois Frances Selfridge (née Baxter) in Ripon, Wisconsin, on 11 January 1858, one of three boys. | wikipage: John Wanamaker text: John Wanamaker John Wanamaker (July 11, 1838December 12, 1922) was an American merchant and religious, civic and political figure, considered by some to be a proponent of advertising and a "pioneer in marketing". He was born in Philadelphia, Pennsylvania, and served as U.S. Postmaster General. Wanamaker was born on July 11, 1838, in a then-rural, unincorporated area that would in time come to be known as the Grays Ferry neighborhood of South Philadelphia. His parents were John Nelson Wanamaker, a brickmaker and a native of Kingwood, New Jersey and Elizabeth Deshong Kochersperger, daughter of a farmer and innkeeper at Gray\'s Ferry whose ancestors had hailed from Rittershoffen in Alsace, France, and from Canton of Bern in Switzerland. In 1860 John Wanamaker married Mary Erringer Brown (18391920). They had six children (two of them died in childhood): John Wanamaker\'s son, Thomas B., who specialized in store financial matters, purchased a Philadelphia newspaper called "The North American" in 1899 and irritated his father by giving regular columns to radical intellectuals such as single-taxer Henry George, Jr., socialist Henry John Nelson (who later became Emma Goldman\'s lawyer), and socialist Caroline H. Pemberton.', 'question: What is the name of the pirate in spongebob? context: wikipage: SpongeBob SquarePants text: Puff endures one of SpongeBob\'s crashes or is otherwise frightened, she puffs up into a ball. Special episodes of the show are hosted by a live action pirate named Patchy and his pet parrot Potty, whose segments are presented in a dual narrative with the animated stories. Patchy is portrayed as the president of a fictional "SpongeBob" fan club, and his greatest aspiration is to meet SpongeBob himself. Potty likes to make fun of Patchy\'s enthusiasm and causes trouble for him while he tries to host the show. An unseen figure called the French Narrator often introduces episodes and narrates the intertitles as if the series was a nature documentary about the ocean. His role and distinctive manner of speaking are references to the oceanographer Jacques Cousteau. Other recurring characters appear throughout the series, such as the muscular lifeguard of Goo Lagoon, Larry the Lobster; a pirate specter known as the Flying Dutchman; and retired superheroes Mermaid Man and Barnacle Boy, who are idolized by SpongeBob and Patrick. The series primarily takes place in the benthic underwater city of Bikini Bottom, which is located in the Pacific Ocean beneath the real-life coral reef known as Bikini Atoll. | wikipage: The SpongeBob Movie: Sponge Out of Water text: The SpongeBob Movie: Sponge Out of Water The SpongeBob Movie: Sponge Out of Water is a 2015 American 3D live-action/animated absurdist comedy film based on the animated television series "SpongeBob SquarePants". A stand-alone sequel to "The SpongeBob SquarePants Movie" (2004), it was directed by former series showrunner Paul Tibbitt in his directorial debut, with live-action sequences directed by Mike Mitchell. It was the first film to be produced by Paramount Animation and the second film in the "SpongeBob SquarePants" film series. The film stars Antonio Banderas and features the show\'s regular voice cast, who returned to reprise their respective roles from the series and the previous film. This movie takes place during the ninth season of "SpongeBob SquarePants". The plot follows a pirate named Burger Beard, who steals the Krabby Patty secret formula using a magical book that makes any text written upon it come true. SpongeBob and his friends must travel to the surface to confront Burger Beard and get the formula back. The film was written by Jonathan Aibel and Glenn Berger from a story conceived by "SpongeBob SquarePants" creator Stephen Hillenburg and Tibbitt. Like the first film, the final act places the animated characters in a live-action world.', 'question: Who plays santa claus in rise of the guardians? context: wikipage: Rise of the Guardians text: "Rise of the Guardians" grossed $103,412,758 in North America, and $203,528,912 in other countries, for a worldwide total of $306,941,670. In North America, the film opened to $32.3 million over its extended five-day weekend, and with $23.8 million over the three-day weekend, it reached fourth place behind "", "Skyfall", and "Lincoln". The film\'s opening was the lowest debut for a DreamWorks Animation film since "Flushed Away". While the film did gross more than double of its $145 million budget, it still did not turn a profit for DreamWorks Animation due to its high production and marketing costs, forcing the studio to take an $83 million write-down. This marked the first time that the studio had lost money on an animated film since "". As a result of this combined with other factors, in February 2013, the studio announced it was laying off 350 employees as part of a company-wide restructuring. The Rome Film Festival and "Vanity Fair" magazine awarded the new Vanity Fair International Award for Cinematic Excellence in November 2012 to "Rise of the Guardians". The film also received the Hollywood Animation Award at the 16th Annual Hollywood Film Festival, held on October 22, 2012. | wikipage: Rise of the Guardians: The Video Game text: Rise of the Guardians: The Video Game Rise of the Guardians is an action-adventure video game (with role playing elements) based on the film of the same name. It is developed by Torus Games and published by D3 Publisher. The game was released on November 20, 2012 in North America and November 23, 2012 in Europe for PlayStation 3, Xbox 360, Wii, Wii U, Nintendo DS, and Nintendo 3DS. The player is able to play as Jack Frost with the help of Santa Claus, the Tooth Fairy, the Easter Bunny, and the Sandman as they battle the evil Pitch Black and his Nightmare minions in order to restore world belief in the Guardians. The game features drop-in/drop-out cooperative play for up to four players, as well as a levelling system that allows the player to unlock greater attacks and special team moves. The game received negative reviews from critics with Metacritic giving it a 43/100 for the Xbox 360 version and a 48/100 for the Wii U version. | wikipage: Fred Tatasciore text: Fred Tatasciore Frederick Tatasciore (; born June 15, 1967) is an American voice actor. Tatasciore has portrayed mostly secondary characters as well as monster-looking types. He is known for voicing the Hulk in several Marvel projects, including the "Marvel Animated Features", "", "" as well as "Avengers Assemble". In video games, he is known for voicing Saren Arterius in the critically acclaimed "Mass Effect" series, Damon Baird in the "Gears of War" series and Zeratul from the game "". He also voices the character "8" in the Tim Burton-produced film "9". His most recent roles are Neftin Prog in "", Nikolai Belinski in the "Call of Duty" franchise, Megatron in "", "" and "", Tookit in "ThunderCats" and the Business Cat in the web series "Our New Electrical Morals", with episodes posted in the Cartoon Hangover YouTube page, administered by Frederator Studios. He has also voiced Snap Shot, Slam Bam, Warnado, Zook and Cuckoo Clocker in the "Skylanders" franchise. Additionally, he voices in Blizzard\'s first-person shooter "Overwatch" and Xür in Bungie\'s first-person shooter, "Destiny".', "question: Where is the u21 euro championships being held? context: wikipage: 2019 UEFA European Under-21 Championship text: 2019 UEFA European Under-21 Championship The 2019 UEFA European Under-21 Championship (also known as UEFA Under-21 Euro 2019) will be the 22nd edition of the UEFA European Under-21 Championship (25th edition if the Under-23 era is also included), the biennial international youth football championship organised by UEFA for the men's under-21 national teams of Europe. The final tournament will be hosted by Italy (and some matches by San Marino) in mid-2019, after their bid was selected by the UEFA Executive Committee on 9 December 2016 in Nyon, Switzerland. A total of 12 teams will play in the tournament, with players born on or after 1 January 1996 eligible to participate. Same as previous Under-21 Championships that were held one year prior to the Olympics, this tournament will serve as European qualifying for the Olympic football tournament, with the top four teams of the tournament qualifying for the 2020 Summer Olympic men's football tournament in Japan, where they will be represented by their under-23 national teams with maximum of three overage players allowed. For the first time, the video assistant referee (VAR) system will be used at the UEFA European Under-21 Championship. Germany are the defending champions. The Italian Football Federation confirmed that Italy would bid to host the tournament in 2019, which also involved the San Marino Football Federation. | wikipage: 2017 UEFA European Under-21 Championship text: 2017 UEFA European Under-21 Championship The 2017 UEFA European Under-21 Championship (also known as UEFA Under-21 Euro 2017) was the 21st edition of the UEFA European Under-21 Championship, a biennial international youth football championship organised by UEFA for the men's under-21 national teams of Europe. The final tournament was hosted in Poland for the first time, after their bid was selected by the UEFA Executive Committee on 26 January 2015 in Nyon, Switzerland. The tournament took place from 16–30 June 2017. Players born on or after 1 January 1994 were eligible for the tournament. In March 2012, UEFA announced that the competition would take place in even numbered years from 2016 onwards. In September 2013, UEFA announced its intention to continue holding the final tournament in odd numbered years following a request from its member national football associations. On 24 January 2014, UEFA confirmed that the final tournament would be held in 2017 and that it would be expanded from 8 teams to 12. The hosts were announced at a meeting of the UEFA Executive Committee in Nyon on 26 January 2015. In late April 2014 the Polish football association PZPN very strongly indicated the country has high chances to host the tournament. Bidding to welcome Europe's best youth teams was one of the reasons for Poland's withdrawal from the UEFA Euro 2020 race. | wikipage: 2015 UEFA European Under-21 Championship text: 2015 UEFA European Under-21 Championship The 2015 UEFA European Under-21 Championship was the 20th edition of the UEFA European Under-21 Championship, a biennial international football competition for men's under-21 national teams organised by UEFA. The final tournament was hosted for the first time in the Czech Republic from 15–30 June 2015, after their bid was selected by the UEFA Executive Committee on 20 March 2012 in Istanbul. Players born on or after 1 January 1992 were eligible to participate in the competition. Fifty-two teams participated in a qualification tournament, taking place between March 2013 and October 2014, to determine the seven teams that would join the final tournament hosts. Holders Spain were not able to defend their title after being eliminated in the qualification play-offs by Serbia. In the final, played at the Eden Arena in Prague, Sweden defeated Portugal 4–3 in a penalty shootout, after a goalless draw at the end of extra-time. In doing so, the Swedish team won their first title in this competition, having previously lost the 1992 final, and secured their first-ever title in UEFA youth competitions on the men's side. By reaching the semi-finals, Denmark, Germany, Portugal and Sweden also qualified for the 2016 Summer Olympics men's football tournament in Brazil.", 'question: Who did she\'s got betty davis eyes? context: wikipage: Bette Davis Eyes text: The song was also a number one hit in 21 countries and peaked at number 10 in the United Kingdom, her only Top 40 hit there to date. According to producer Val Garay, the original demo of the tune that was brought to him sounded like "a Leon Russell track, with this beer-barrel polka piano part." The demo can be heard in a Val Garay interview on TAXI TV at 21:50. Keyboardist Bill Cuomo came up with the signature synth riff, using the Sequential Circuits Prophet-5 synthesizer, which now defines Carnes\'s version. The song was recorded in the studio on the first take. Bette Davis, then 73 years old, wrote letters to Carnes, Weiss, and DeShannon to thank all three of them for making her "a part of modern times," and said her grandson now looked up to her. After their Grammy wins, Davis sent them roses as well. The song was ranked at number 12 on "Billboard\'s" list of the top 100 songs in the first 50 years of the "Billboard" Hot 100 chart. Cleopatra Records released a re-recording of the song as a single in 2007. The video, directed by Australian film director Russell Mulcahy, received heavy airplay when it premiered. The video stars with a leaning figure draped in black at the center of a dance hall. | wikipage: Bette Davis Eyes text: Bette Davis Eyes "Bette Davis Eyes" is a song written and composed by Donna Weiss and Jackie DeShannon, and made popular by American singer Kim Carnes. DeShannon recorded it in 1974; Carnes\'s 1981 version spent nine weeks at No. 1 on the "Billboard" Hot 100 and was "Billboard"s biggest hit of 1981. The song was written in 1974 by Donna Weiss and Jackie DeShannon. DeShannon recorded the song that same year on her album "New Arrangement." In this original incarnation, the track is performed in an "R&B lite" arrangement, featuring a prominent uptempo piano part, as well as flourishes of pedal steel guitar and horns. However, it was not until 1981, when Kim Carnes recorded her version of the song in a radically different synthesizer-based arrangement, that "Bette Davis Eyes" became a commercial success. The Carnes version spent nine non-consecutive weeks on top of the US "Billboard" Hot 100 (interrupted for one week by the "Stars on 45 Medley") and was "Billboard"s biggest hit of the year for 1981. The single also reached No. 5 on Billboard\'s Top Tracks charts and No. 26 on the Dance charts. The song won the Grammy Awards for Song of the Year and Record of the Year.', 'question: Who developed an explanation for the photoelectric effect? context: wikipage: Photoelectric effect text: For surface states and molecules the three-step model does still make some sense as even most atoms have multiple electrons which can scatter the one electron leaving. When a surface is exposed to electromagnetic radiation above a certain threshold frequency (typically visible light for alkali metals, near ultraviolet for other metals, and extreme ultraviolet for non-metals), the radiation is absorbed and electrons are emitted. Light, and especially ultra-violet light, discharges negatively electrified bodies with the production of rays of the same nature as cathode rays. Under certain circumstances it can directly ionize gases. The first of these phenomena was discovered by Hertz and Hallwachs in 1887. The second was announced first by Philipp Lenard in 1900. The ultra-violet light to produce these effects may be obtained from an arc lamp, or by burning magnesium, or by sparking with an induction coil between zinc or cadmium terminals, the light from which is very rich in ultra-violet rays. Sunlight is not rich in ultra-violet rays, as these have been absorbed by the atmosphere, and it does not produce nearly so large an effect as the arc-light. Many substances besides metals discharge negative electricity under the action of ultraviolet light: lists of these substances will be found in papers by G. C. Schmidt and O. Knoblauch. | wikipage: Photoelectric effect text: Above the threshold frequency, the maximum kinetic energy of the emitted photoelectron depends on the frequency of the incident light, but is independent of the intensity of the incident light so long as the latter is not too high. For a given metal and frequency of incident radiation, the rate at which photoelectrons are ejected is directly proportional to the intensity of the incident light. An increase in the intensity of the incident beam (keeping the frequency fixed) increases the magnitude of the photoelectric current, although the stopping voltage remains the same. The time lag between the incidence of radiation and the emission of a photoelectron is very small, less than 10 second. The direction of distribution of emitted electrons peaks in the direction of polarization (the direction of the electric field) of the incident light, if it is linearly polarized. In 1905, Einstein proposed an explanation of the photoelectric effect using a concept first put forward by Max Planck that light waves consist of tiny bundles or packets of energy known as photons or quanta. The maximum kinetic energy formula_1 of an ejected electron is given by where formula_3 is the Planck constant and formula_4 is the frequency of the incident photon. | wikipage: Photoelectric effect text: To make sense of the fact that light can eject electrons even if its intensity is low, Albert Einstein proposed that a beam of light is not a wave propagating through space, but rather a collection of discrete wave packets (photons), each with energy "h". This shed light on Max Planck\'s previous discovery of the Planck relation () linking energy ("E") and frequency ("") as arising from quantization of energy. The factor "h" is known as the Planck constant. In 1887, Heinrich Hertz discovered that electrodes illuminated with ultraviolet light create electric sparks more easily. In 1900, while studying black-body radiation, the German physicist Max Planck suggested that the energy carried by electromagnetic waves could only be released in "packets" of energy. In 1905, Albert Einstein published a paper advancing the hypothesis that light energy is carried in discrete quantized packets to explain experimental data from the photoelectric effect. This model contributed to the development of quantum mechanics. In 1914, Millikan\'s Experiment supported Einstein\'s model of the photoelectric effect. Einstein was awarded the Nobel Prize in 1921 for "his discovery of the law of the photoelectric effect", and Robert Millikan was awarded the Nobel Prize in 1923 for "his work on the elementary charge of electricity and on the photoelectric effect".'],
#    'prompts_input_ids': torch.tensor([[ 822,   10, 2645,    0,    0,    0],
#        [ 822,   10,  363,    0,    0,    0],
#        [ 822,   10, 2645,    0,    0,    0],
#        [ 822,   10, 2840,    0,    0,    0],
#        [ 822,   10, 2645,    0,    0,    0],
#        [ 822,   10, 2645,    0,    0,    0]], device='cuda:0'),
#    'prompts_attention_mask': torch.tensor([[1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0]], device='cuda:0'),
#    'generated_text': ['Kung Fu Panda is a media franchise by DreamWorks Animation, consisting of three films: Kung Fu Panda (2008), Kung Fu Panda 2 (2011) and Kung Fu Panda 3 (2016). Angelina Jolie plays the role of Tigress in Kung Fu Panda: Secrets of the Scroll, while Jackie Chan plays the role of Monkey.', 'The Middle Ages was a period of European history lasting from around the 5th century to the 15th century. The Enlightenment was an intellectual and philosophical movement that dominated the world of ideas in Europe during the 18th century, the "Century of Philosophy". The early modern period of modern history follows the late Middle Ages of the post-classical era.', '"The customer is always right" is a motto or slogan which exhorts service staff to give a high priority to customer satisfaction. It was popularised by pioneering and successful retailers such as Harry Gordon Selfridge, John Wanamaker and Marshall Field. They advocated that customer complaints should be treated seriously so that customers do not feel cheated or deceived. This attitude was novel and influential when misrepresentation was rife and "caveat emptor" (let the buyer beware) was a common legal maxim.', "The SpongeBob SquarePants television series is an American children's television series created by Paramount Animation. In the show, the pirate is named Burger Beard. In the movie, the pirate is named Puff.", 'Rise of the Guardians is an American animated film produced by DreamWorks Animation. American actor Fred Tatasciore voices Santa Claus in the film. American actor Michael J. Fox voices Santa Claus in the video game Rise of the Guardians.', "The UEFA European Under-21 Championship is a biennial international youth football competition for men's under-21 national teams organised by UEFA. The 2015 UEFA European Under-21 Championship was the 20th edition of the UEFA European Under-21 Championship. The final tournament was hosted for the first time in the Czech Republic from 15–30 June 2015. The 2017 UEFA European Under-21 Championship was the 21st edition of the UEFA European Under-21 Championship, a biennial international youth football competition for men's under-21 national teams of Europe. The 2017 UEFA European Under-21 Euro Championship was held in Nyon, Switzerland.", '"Bette Davis Eyes" is a song written and composed by Donna Weiss and Jackie DeShannon, and made popular by American singer Kim Carnes. DeShannon recorded the song in 1974, while Bette Davis wrote letters to all three of them to thank them for making her "a part of modern times." The original incarnation of the track is performed in an "R&B lite" arrangement, featuring a prominent uptempo piano part, as well as flourishes of pedal steel guitar and horns. However, it was not until 1981, when Kim Carnes recorded her version in a radically different synthesizer-based arrangement, that "Billboard" became a commercial success.', "In 1905, Albert Einstein proposed an explanation of the photoelectric effect using a concept first put forward by Max Planck that light waves consist of tiny bundles or packets of energy known as photons or quanta. This model contributed to the development of quantum mechanics. In 1914, Millikan's Experiment supported Einstein's model of the photoelectric effect."],
#    'generated_input_ids': torch.tensor([[ 480,  425, 6343,    0,    0,    0],
#        [  37, 4551, 7526,    0,    0,    0],
#        [  96,  634,  884,    0,    0,    0],
#        [  37,    3, 5078,    0,    0,    0],
#        [  96,  279, 1954,    0,    0,    0],
#        [  86,  957, 3076,    0,    0,    0]], device='cuda:0'),
#    'generated_attention_mask': torch.tensor([[1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0],
#        [1, 1, 1, 0, 0, 0]], device='cuda:0')
#}
#print('dooooge1')
#print(type(test_raw_data), flush=True)
#test_raw_data = accelerator.gather_for_metrics(test_raw_data)
#print('dooooge2')
#print(type(test_raw_data), flush=True)
#print(test_raw_data, flush=True)
#print(test_raw_data['generated_input_ids'], flush=True)
#input()
#exit()

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="path to config file")
args = parser.parse_args()
# load yaml file
with open(args.config) as f:
    args =yaml.safe_load(f)


# prepare data
class TextGenDataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None):
        super().__init__()
        
        self.split = split
        self.dataset_fns = {
            "train": "../data/fine_tuning_data/train.json",
            "dev":   "../data/fine_tuning_data/dev.json",
            "test":  "../data/fine_tuning_data/test.json"
        }
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:
            instances.append({
                "prompt": task_instance['text'],
                "metadata": {
                    "prompt": task_instance['text'],
                    "references": task_instance['answer'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],}
            })
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata
        }
        return result
    

def main():

    # set seed
    set_seed(args['train']['seed'], args['train']['cuda_deterministic'])
    
    # set saving directories
    log_info(f"Write to output directory: {args['logging']['save_dir']}")
    
    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        # save the config file
        with open(os.path.join(args['logging']['save_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    
    # initialize policy and value model tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                           model_max_length=args['env']['max_input_len'])
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    
    
    # Load data
    log_info(f'Loading data ...')
    train_dataset = TextGenDataset( 'train', tokenizer, accelerator=accelerator)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                  shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    eval_dataset = TextGenDataset( 'dev',  tokenizer, accelerator=accelerator, length_limit=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)


    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    ref_policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    
    value = T5Value(
        model_ckpt=args['model']['value_model']['ckpt'],
        model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
        tokenizer=tokenizer,
        accelerator=accelerator,
        freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
        )
    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)
    
    reward = FineGrainedCompactReward(
        tokenizer=tokenizer,
        factual_model_ckpt=args['reward']['factuality_compact_model']['ckpt'],
        kl_coef=args['ppo']['kl_coef'],
        factuality_positive_reward = args['reward']['factuality_compact_model']['positive_reward'],
        factuality_negative_reward = args['reward']['factuality_compact_model']['negative_reward'],
        sep = "</s>"
    )
    
    # prepare reward models
#    reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
    reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
#    reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)
    
    # prepare optimizers and schedulers
    if args['model']['value_model']['policy_value_sharing']:
        parameters = chain(policy.model.parameters(), policy.linear.parameters())
    else:
        parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters(), value.linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args['train']['lr'], eps=1e-5)
    total_steps = ceil_div(args['train']['total_episodes'], 
                                args['train']['sampling_batch_size_per_card'] * accelerator.num_processes * args['env']['train_num_samples_per_input'])
    
    scheduler = transformers.get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=100*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes,
        num_training_steps=total_steps*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes,
    )
    
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)


    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        log_info=log_info,
    )
    
    steps = list(range(total_steps + 1))
    steps = tqdm(steps) if accelerator.is_main_process else steps
    for step in steps:
        trainer.train(step)
        accelerator.wait_for_everyone()
        # early stopping because KL explodes
        if trainer.should_early_stop:
            if accelerator.is_local_main_process:
                print("Early stopping triggered. Terminating training.")
            break
            
if __name__ == '__main__':
    main()

