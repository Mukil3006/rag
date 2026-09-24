"""Only the verifier's grounded excerpts cross the proposed system's generation boundary."""
def apply_gate(rows,comparison):
    warnings=[]
    blocked=[]
    approved=[]
    conflict_ids={c[side+'_id'] for c in comparison.get('conflicts',[]) for side in ('left','right')}
    conflict_error=any(r['chunk_id'] in conflict_ids and r['verification'].get('error') for r in rows)
    if comparison.get('error') or conflict_error:
        return {'status':'INSUFFICIENT','approved_evidence':[],'blocked_ids':[r['chunk_id'] for r in rows],
                'warnings':['Cross-evidence verification failed; no evidence released.'], 'conflicts':[], 'error':comparison.get('error','A conflict counterpart failed individual verification.')}
    for row in rows:
        v=row['verification']
        if v['verdict']=='INSUFFICIENT' or v.get('error'):
            blocked.append(row['chunk_id'])
            if v.get('error'): warnings.append('Verification failed for '+row['chunk_id'])
            continue
        if not v['supporting_claims']:
            blocked.append(row['chunk_id'])
            continue
        approved.append({'chunk_id':row['chunk_id'],'child_id':row.get('child_id',row['chunk_id']),
                         'source':row['source'],'page':row['page'],'parent_id':row['parent_id'],
                         'parent_expansion':row['parent_expansion'],'text':'\n'.join(v['supporting_claims']),
                         'verification':v})
        if v['verdict']=='PARTIAL': warnings.append('Partial evidence: '+row['chunk_id']+'; missing information must not be inferred.')
    verdicts={r['verification']['verdict'] for r in approved}
    status=('CONTRADICTORY' if 'CONTRADICTORY' in verdicts else 'SUPPORTED' if 'SUPPORTED' in verdicts else 'PARTIAL' if approved else 'INSUFFICIENT')
    if status=='CONTRADICTORY': warnings.append('Sources disagree. Report both claims with citations; do not choose a winner.')
    if status!='CONTRADICTORY' and comparison.get('comparison_scope') and comparison.get('verified_source_count',0)<2 and approved:
        status='PARTIAL'
        warnings.append('Only one source supplies a relevant fact; agreement between sources cannot be established.')
    return {'status':status,'approved_evidence':approved,'blocked_ids':blocked,'warnings':warnings,'conflicts':comparison['conflicts']}
