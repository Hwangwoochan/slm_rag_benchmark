import json
import pandas as pd

def summarize_rag_performance(file_path):
    all_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                metadata = data.get('metadata', {})
                
                all_data.append({
                    "Model": metadata.get('model', 'Unknown'),
                    "Chunk": metadata.get('chunk_config', 'Unknown'),
                    "Score": data.get('prometheus_score', 0)
                })
        
        df = pd.DataFrame(all_data)
        
        # 1. 모델과 청킹으로 그룹화하여 평균 점수 및 개수 계산
        summary_df = df.groupby(['Model', 'Chunk']).agg(
            Avg_Score=('Score', 'mean'),
            Count=('Score', 'count')
        ).reset_index()
        
        # 점수 내림차순 정렬
        summary_df = summary_df.sort_values(by='Avg_Score', ascending=False)
        
        return summary_df

    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")
        return None

# --- 실행 ---
file_name = "rag_techqa_llm_judge_data_W_eval_results.jsonl"  # 실제 파일명
result_table = summarize_rag_performance(file_name)

if result_table is not None:
    print("### [모델/청킹 설정별 성능 요약] ###")
    print(result_table.to_markdown(index=False))

