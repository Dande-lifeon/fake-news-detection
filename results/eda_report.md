# EDA: ISOT Fake and Real News

## 1. Overview

- rows: **44,898**
- columns: id, title, text, subject, date, label, source
- real (label=0): **21,417**
- fake (label=1): **23,481**

### Missing values

| column | missing |
|---|---|
| title | 0 |
| text | 0 |
| subject | 0 |
| date | 0 |

- empty body: **631**

## 2. Subject × label (label-leak check)

| subject | real | fake | total |
|---|---|---|---|
| Government News | 0 | 1570 | 1570 |
| Middle-east | 0 | 778 | 778 |
| News | 0 | 9050 | 9050 |
| US_News | 0 | 783 | 783 |
| left-news | 0 | 4459 | 4459 |
| politics | 0 | 6841 | 6841 |
| politicsNews | 11272 | 0 | 11272 |
| worldnews | 10145 | 0 | 10145 |
| total | 21417 | 23481 | 44898 |

- subjects only in real: ['politicsNews', 'worldnews']
- subjects only in fake: ['Government News', 'Middle-east', 'News', 'US_News', 'left-news', 'politics']
- shared subjects: none
- `subject` is audit-only and must never enter a model feature.

## 3. Dates

- parseable dates: 44,888 / 44,898
- range: 2015-03-31 → 2018-02-19

| month | real | fake |
|---|---|---|
| 2017-02 | 591 | 879 |
| 2017-03 | 684 | 1066 |
| 2017-04 | 543 | 805 |
| 2017-05 | 505 | 827 |
| 2017-06 | 556 | 814 |
| 2017-07 | 540 | 777 |
| 2017-08 | 657 | 795 |
| 2017-09 | 3074 | 636 |
| 2017-10 | 3172 | 605 |
| 2017-11 | 3096 | 523 |
| 2017-12 | 2533 | 447 |
| 2018-02 | 0 | 35 |

## 4. Text lengths (characters, after cleaning)

| index | label | field | n | mean | median | p95 | empty |
|---|---|---|---|---|---|---|---|
| 0 | real | title | 21417 | 64.7 | 64.0 | 80.0 | 0 |
| 1 | real | text | 21417 | 2374.2 | 2213.0 | 5499.2 | 1 |
| 2 | fake | title | 23481 | 93.8 | 89.0 | 147.0 | 0 |
| 3 | fake | text | 23481 | 2533.0 | 2155.0 | 5598.0 | 630 |

## 5. Duplicates

- exact duplicate rows (title+body): **5,798**
- cross-label conflict keys: **0**

## 6. Class balance

- fake share: **0.5230** (distance from 0.5: 0.0230)
