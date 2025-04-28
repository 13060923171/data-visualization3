phase1_y2 = df5[(df5['时间'] == '2023-08-24')]
list_y2.append(phase1_y2)
phase2_y2 = df5[(df5['时间'] >= '2023-08-25') & (df5['时间'] <= '2023-09-30')]
list_y2.append(phase2_y2)
phase3_y2 = df5[(df5['时间'] >= '2023-10-01') & (df5['时间'] <= '2024-02-28')]
list_y2.append(phase3_y2)
phase4_y2 = df5[(df5['时间'] >= '2024-03-01') & (df5['时间'] <= '2024-07-31')]
list_y2.append(phase4_y2)