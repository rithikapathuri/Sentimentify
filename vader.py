from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#with open("test1.txt") as file: 
 #   lines = file.readlines()

#lines = ' '.join(lines)

prompt = """
   "Why Coffee Drinkers Are Clearly Superior to Tea Drinkers"

It is an undeniable fact that coffee drinkers are simply more intelligent, more productive, and more cultured than those who opt for the weak, uninspiring beverage known as tea. While coffee fuels the world's greatest minds—scientists, entrepreneurs, and artists—tea drinkers are often found aimlessly sipping their lukewarm leaf water, contributing little to society.

Studies (which we won’t bother citing) have clearly shown that coffee enhances brain function, while tea only serves to lull people into a false sense of sophistication. Every major innovation of the past century? Powered by coffee. Tea, on the other hand, has been the beverage of indecision and mediocrity, embraced by those who prefer inactivity over action.

Let’s be real—when was the last time you heard someone say, “I need a cup of tea to power through this work”? Never. Because it doesn’t happen. Tea drinkers are just delaying the inevitable realization that coffee is king and the only true beverage of champions.
"""

prompt2 = """

ATLANTA (AP) — Donald Trump’s second administration has put forth an avalanche of policy changes and political pronouncements that have jolted Washington and the world.

That agenda is taken largely from his “Agenda 47” campaign proposals, the Heritage Foundation’s Project 2025 and other hard-right influencers with juice in Trump’s White House. There is much more, however, that the president and those groups discussed on the campaign trail but have yet to attempt.

Here’s a look at some substantial proposals still pending.

Shuttering the Department of Education
The right has long targeted the Department of Education, which became a Cabinet agency in 1980 under President Jimmy Carter. Trump aides have prepared an executive order that would limit if not effectively shut down the department.

“I want Linda to put herself out of a job,” Trump said of Education Secretary-designee Linda McMahon, who awaits Senate confirmation.

The timing, though, remains uncertain as the White House grapples with how to unwind an agency that was established by law and involves billions in spending approved by Congress, including Title I money for low-income schools and college student loans.


Tightening restrictions on abortion pills and other actions
Trump sidestepped and obfuscated on abortion during the campaign. He bragged that his Supreme Court nominees helped overturn the Roe v. Wade precedent and shifted control of abortion restrictions to state governments — but said he would not sign a national ban. Then he changed course and said he would ban abortion later in pregnancy, though he did not specify when that would be.

Project 2025 proposes a range of ideas, most of which would come under the purview of Robert F. Kennedy Jr. if the Senate confirms him as Health and Human Services secretary:

— It seeks tighter restrictions on abortion pills, demand for which rose after Trump’s election. The document says the administration should revoke the Food and Drug Administration’s approval of medication abortion drugs. Short of that, if the drugs remain on the market, the document urges Trump to “reinstate earlier safety protocols for Mifeprex that were mostly eliminated in 2016 and apply these protocols to any generic version of mifepristone.” Specifically, Project 2025 calls for “a bare minimum” deadline of the 49th day of gestation for dispensing the drugs (it is now 70 days), requiring in-person dispensing, and requiring prescribers to report “all serious adverse events, not just deaths.” During his confirmation hearings last month, Kennedy said Trump has asked him to study mifepristone, a drug used to terminate pregnancies and help women complete miscarriages.

— If those paths are not sufficient to limit medication abortions, it proposes invoking an 1873 anti-obscenity law, the Comstock Act, as justification to block the mailing of any abortion-related materials. When asked during an April 12, 2024, interview with Time magazine for his views on the Comstock Act and the mailing of abortion pills, Trump promised to make a statement on the issue in the next 14 days, saying: “I feel very strongly about it. I actually think it’s a very important issue.” He never made that statement.
"""

prompt3 = """
CNN mixed up the names of former President Barack Obama and Osama bin Laden in an embarrassing on-air gaffe on Friday night.

During a segment on the Guantanamo Bay detention camp on "CNN News Central," a graphic appeared behind anchor Boris Sanchez that read, "OBAMA BIN LADEN ASSOCIATE: ABU ZUBAYDAH."

Abu Zubaydah – the suspected Palestinian terrorist whose real name is Zayn al-Abidin Muhammad Husayn – is currently being held at Guantanamo Bay, Cuba, while authorities are investigating the circumstances surrounding his apprehension and detention.

When President Obama took office in January 2009, he pledged to close the detention facility within a year but failed to do so. This failure enabled future presidents, like President Donald Trump, to use the detention facility to house some of the world's most dangerous criminals.

Earlier this week, Trump instructed the Pentagon to prepare the facility to hold around 30,000 'criminal illegal aliens' at the US military base.

While Sanchez did not read "Obama bin Laden" out loud, viewers quickly caught the glaring mashup of names, and took to social media to call it out.

"So it's confirmed now. The Babylon Bee bought CNN and just didn't tell us," X user Charles Jorhan mocked.
"""
sid_obj = SentimentIntensityAnalyzer()
sentiment_dict = sid_obj.polarity_scores(prompt3)
print(sentiment_dict)