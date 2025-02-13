import re

q = "select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( select substring(c_phone from 1 for 2) as cntrycode, c_acctbal from customer where substring(c_phone from 1 for 2) in ('41', '35', '25', '27', '37', '30', '26') and c_acctbal > ( select avg(c_acctbal) from customer where c_acctbal > 0.00 and substring(c_phone from 1 for 2) in ('41', '35', '25', '27', '37', '30', '26') ) and not exists ( select * from orders where o_custkey = c_custkey ) ) as custsale group by cntrycode order by cntrycode LIMIT 1;"
#q = 'select foo from bar where baz = "wow"'

def extract_subqueries(query, recurse = False):
    # nightmarish lol
    REGEX = '\\(\\s*(SELECT.*)[^\\(\\)]*(?:(?:avg|sum|count|substring)\\([^\\(\\)]*\\)*)*\\s+(?:from|where|and|group|order)'
    subqueries = []

    matches = re.findall(REGEX, query, re.IGNORECASE)
    subqueries.append(re.sub(REGEX, '( SUBQUERY )', query, flags=re.IGNORECASE))
    found_sqs = [x for x in matches]

    if found_sqs:
        subqueries.extend(found_sqs)
        for sq in found_sqs:
            subqueries.extend(extract_subqueries(sq, True))
        return subqueries
    else:
        return [query] if not recurse else []


print(extract_subqueries(q))