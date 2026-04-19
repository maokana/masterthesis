#!/usr/bin/python3
import csv
import itertools

def csvread(fp):
  with open(fp) as f:
    reader = csv.reader(f)
    return [row for row in reader]

def jaccard_similarity_coefficient(list_a,list_b):
    #集合Aと集合Bの積集合(set型)を作成
    set_intersection = set.intersection(set(list_a), set(list_b))
    #集合Aと集合Bの積集合の要素数を取得
    num_intersection = len(set_intersection)
 
    #集合Aと集合Bの和集合(set型)を作成
    set_union = set.union(set(list_a), set(list_b))
    #集合Aと集合Bの和集合の要素数を取得
    num_union = len(set_union)
 
    #積集合の要素数を和集合の要素数で割って
    #Jaccard係数を算出
    try:
        return float(num_intersection) / num_union
    except ZeroDivisionError:
        return 1.0 

def main():
  rows = csvread('対象分野.csv')
  #print(rows)
  taisho_list = []
  taisho_journal_list = []
  for row in rows:
    #print(row) 
    row_trimed = [a for a in row if a != '']
    #print(row_trimed)
    taisho_journal = row_trimed[0]
    sanka_journal = row_trimed[1:]
    row_source = [taisho_journal, sanka_journal]
    #print(row_source)
    taisho_list.append(row_source)
    taisho_journal_list.append(taisho_journal)
  #print(taisho_list)
  taisho_journal_list.sort()
  #print(taisho_journal_list)
  combi_list = list(itertools.combinations(taisho_journal_list, 2))
  #print(combi_list)

  for combi in combi_list:
    list_a_key = combi[0]
    list_b_key = combi[1]
    #print(list_a_key, list_b_key)
    for l in taisho_list:
      if list_a_key in l[0]:
        list_a = l[1]
        break

    for l in taisho_list:
      if list_b_key in l[0]:
        list_b = l[1]
        break
    #print(list_a_key, list_a, list_b_key, list_b)
    #print(jaccard_similarity_coefficient(list_a, list_b))
    jaccard_value = jaccard_similarity_coefficient(list_a, list_b)
    #list_a_title = 'Frontiers_in_{}.txt'.format(list_a_key)
    #list_b_title = 'Frontiers_in_{}.txt'.format(list_b_key)
    title_vs = 'Frontiers_in_{}.txt vs Frontiers_in_{}.txt'.format(list_a_key, list_b_key)
    result = '{}, {}'.format(title_vs, jaccard_value)
    print(result)

if __name__ == '__main__':
  main()

