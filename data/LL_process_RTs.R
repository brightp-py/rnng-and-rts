rm(list=ls())
library(tidyverse)

b1 = read_csv("data/batch1_pro.csv")
b2 = read_csv("data/batch2_pro.csv")
words = read_tsv("data/all_stories.tok")

b = bind_rows(b1, b2)

offset = 230

## me messing with stuff



df_old <- b %>%
  inner_join(words) %>%
  filter(RT > 100, RT < 3000, correct > 4) %>%
  arrange(WorkerId, item, zone) %>% 

df_new <- b %>%
  filter(!(item == 3 & zone == offset + 1)) %>%
  mutate(zone=if_else(item == 3 & zone > offset, zone - 3, zone - 2)) %>%
  inner_join(words) %>%
  filter(RT > 100, RT < 3000, correct > 4) %>%
  arrange(WorkerId, item, zone)

df_3old <- df_old %>% filter(WorkerId == "A117RW2F1MNBQ8", item == 3)
df_3old <- b %>% filter(item == 3) %>% filter(zone == 229) %>% group_by(WorkerId) %>%  count()


## original code down here
d = b %>%
    filter(!(item == 3 & zone == offset + 1)) %>%
    mutate(zone=if_else(item == 3 & zone > offset, zone - 3, zone - 2)) %>%
    inner_join(words) %>%
    filter(RT > 100, RT < 3000, correct > 4) %>%
    arrange(item, zone)

gmean <- function(x) exp(mean(log(x)))
gsd   <- function(x) exp(sd(log(x)))

wordinfo = d %>%
    group_by(word, zone, item) %>%
    summarize(nItem=n(),
              meanItemRT=mean(RT),
              sdItemRT=sd(RT),
              gmeanItemRT=gmean(RT),
              gsdItemRT=gsd(RT)) %>%
    ungroup()

write_tsv(wordinfo, "processed_wordinfo.tsv")

d %>% inner_join(wordinfo) %>% write_tsv("processed_RTs.tsv")



