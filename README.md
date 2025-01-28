# Rotary Positional Embedding: [[paper]](https://arxiv.org/pdf/2104.09864v5.pdf)
* Let
  ![image](https://github.com/user-attachments/assets/e10b2a9d-37d6-467d-b728-7d8f5906a13f)
* We need the transformation functions above ($f_q$ and $f_k$), such that they satisfy the below equation, the resultant function should encode **relative** position information
  ![image](https://github.com/user-attachments/assets/5d9538aa-d3bf-4c25-b2b8-3d1cd78ceb5f)
  ![1.png](rope3.png)
  ![image](https://github.com/user-attachments/assets/8e679770-6483-4cba-b530-36fa5cc90ddc)

* ![2.png](rope2.png)
  ![3.png](rope1.png)


## Also See
* [Reference-1](https://github.com/ZhuiyiTechnology/roformer/tree/main?tab=readme-ov-file#implementation)
* [Reference-2](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L80C1-L161C50)

* ![image](https://github.com/user-attachments/assets/1aca7332-12f1-4155-8183-37b2168d6b51)
* ![image](https://github.com/user-attachments/assets/d260e4e8-02c7-40d8-b863-c915963c7e12)
