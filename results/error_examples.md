# FP/FN examples: LR fusion (test)

Threshold 0.5. Snippets are the first 260 characters of `text_clean`.

## False positives (real predicted fake)

| id | p(fake) | title | snippet |
|---:|---:|---|---|
| 10524 | 0.9599 | Text: Mitt Romney speech slamming Donald Trump | (Reuters) - The following is the full text of the speech delivered on Thursday by former Massachusetts Governor and 2012 Republican nominee for U.S. president Mitt Romney to the Institute of Politics at the University of Utah. I am not here to announce my c... |
| 12581 | 0.8412 | Pictures of the Year: A picture and its story | (Reuters) - Reuters photographers witnessed the biggest stories of 2017 from Donald Trump s first year as U.S. president to the flight of Rohingya refugees from Myanmar, from Venezuela s crisis to the fall of Islamic State and a tower block inferno in Londo... |

## False negatives (fake predicted real)

| id | p(fake) | title | snippet |
|---:|---:|---|---|
| 38638 | 0.0980 | “GREXIT”? GREEKS VOTE NO ON MORE AUSTERITY IN RETURN FOR RESCUE LOANS | A number of European politicians, including Jeroen Dijsselbloem, the top eurozone official, have said a no vote would jeopardize Greece s place in the 19-nation eurozone. Investors are also likely to believe a no win increases the chance of a so-called Grex... |
| 38558 | 0.1950 | FOLLOW THE MONEY: MEXICAN GIVEAWAY OF MILLIONS OF FLAT SCREEN TV’S TO THE POOR RAISES QUESTIONS | Follow the money on this one Mexico is spending over $1.6 billion dollars on flat screens for millions at a cost of $145.00 instead of just $40.00 for a decoder box. I m sure if you dig deep enough you ll find some sort of Carlos Slim corruption. He s the M... |
