#include <bits/stdc++.h>
#define N 100500
using namespace std;

const int num_of_walks = 40;
const int seq_len = 4;
int o[N];
char dis[N][N];
bool vis[N];
int n, m;
vector<int> E[N];

void link(int u, int v) {
    E[u].push_back(v);
}

void bfs(int S) {
    queue<int> q;
    q.push(S);
    dis[S][S] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
	if (dis[S][u] > seq_len) return ;
	for (int i=0;i<(int)E[u].size();i++) {
            int v = E[u][i];
	    if (dis[S][v] == 0) {
                dis[S][v] = dis[S][u] + 1;
		q.push(v);
	    }
	}
    }
    return ; 
}

int main() {
    freopen("edge_input/cornel.in", "r", stdin);
    freopen("preprocess/cornell_40_4_m.txt", "w",stdout);
    srand(time(0));
    scanf("%d%d",&n, &m);


    for (int i=0;i<n;i++) link(i, i);

    for (int i=1;i<=m;i++) {
        int u, v;
        scanf("%d%d",&u, &v);
        if (u==v) continue;
	link(u, v);
        link(v, u);
    }

    for (int i=0;i<n;i++) bfs(i);

    for (int epoch=0;epoch<1000;epoch++) {
        for (int st=0;st<n;st++) {
            for (int i=0;i<num_of_walks;i++) {
                int u = st;
                printf("[");
                for (int _=0;_<seq_len;_++) {
                    printf("%d", u);
                    o[_] = dis[st][u];
		    printf(", ");
                    int tot = E[u].size();
                    int g = rand() % tot;
                    u = E[u][g];    
                }

                for (int _=0;_<seq_len;_++) {
                    printf("%d", o[_]);
		    if (_!=seq_len-1) printf(", ");
		}

                printf("]\n");
            }
        }
    }
    fclose(stdin);
    fclose(stdout);
    return 0;
}
