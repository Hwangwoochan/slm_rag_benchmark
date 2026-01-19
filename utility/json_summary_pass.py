import json
import pandas as pd

def summarize_rag_performance_filtered(file_path):
    all_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # 1. 모델의 답변(predict 필드) 추출
                # 필드명이 'predict'가 맞는지 확인 필요 (만약 다르면 이 부분만 수정)
                prediction = data.get('prediction', '')
                
                # 2. "don't know"가 포함되어 있으면 통계에서 제외 (pass)
                if "don't know" in prediction.lower():
                    continue
                
                # 3. 필터링을 통과한 데이터만 리스트에 추가
                metadata = data.get('metadata', {})
                all_data.append({
                    "Model": metadata.get('model', 'Unknown'),
                    "Chunk": metadata.get('chunk_config', 'Unknown'),
                    "Score": data.get('prometheus_score', 0)
                })
        
        if not all_data:
            print("모든 답변이 필터링되었거나 데이터가 없습니다.")
            return None

        df = pd.DataFrame(all_data)
        
        # 모델과 청킹별 평균 점수 계산
        summary_df = df.groupby(['Model', 'Chunk']).agg(
            Avg_Score=('Score', 'mean'),
            Valid_Count=('Score', 'count') # 'don't know'를 제외한 유효 답변 개수
        ).reset_index()
        
        return summary_df.sort_values(by='Avg_Score', ascending=False)

    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")
        return None

# --- 실행 ---
file_name = "leaf_eval_results.jsonl"
result_table = summarize_rag_performance_filtered(file_name)

if result_table is not None:
    print("### [필터링 완료: 'don't know' 답변 제외 성능 요약] ###")
    print(result_table.to_markdown(index=False))