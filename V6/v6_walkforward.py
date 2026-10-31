"""Leakage-safe expanding evaluation of V6 research-only heuristic.
Each draw is predicted from strictly earlier rows. No AI accuracy claim.
"""
import argparse,csv,json,math
from v6_e2e import load_history,predict

def evaluate(draws):
    counts=[0]*7
    second_hits=0
    records=[]
    for i in range(10,len(draws)):
        guess=predict(draws[:i])
        actual=draws[i]
        hits=len(set(guess["first"])&set(actual["first"]))
        second=int(guess["second"]==actual["second"])
        counts[hits]+=1
        second_hits+=second
        records.append({"draw_id":actual["draw_id"],"first_hits":hits,
                        "second_hit":bool(second),"training_through":draws[i-1]["draw_id"]})
    n=len(records)
    return {"predictions":n,"hit_distribution_0_to_6":counts,
            "first_3plus_rate":sum(counts[3:])/n if n else None,
            "second_hit_rate":second_hits/n if n else None,
            "random_first_3plus_rate":sum(math.comb(6,k)*math.comb(32,6-k) for k in range(3,7))/math.comb(38,6),
            "random_second_hit_rate":1/8,
            "rows":records,"trained_ai_integrated":False,
            "blind_test_performed":False,"predictive_advantage_verified":False,
            "evaluation":"prequential_exploratory_no_future_draws"}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(load_history(a.csv)),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
