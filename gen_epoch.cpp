#include <bits/stdc++.h>
#define N 100050
using namespace std;

const int num_of_walks = 40;
const int seq_len = 5;
int o[N];
char dis[N][N];
bool vis[N];
int n, m;
vector<int> E[N];
char s[1000] = "Electronics_40_5_0000.txt";

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
    freopen("cora.in", "r", stdin);
    srand(time(0));
    scanf("%d%d",&n, &m);

    printf("%d %d\n", n, m);

    for (int i=0;i<n;i++) link(i, i);

    for (int i=1;i<=m;i++) {
        int u, v;
        scanf("%d%d",&u, &v);
        if (u == v) continue;
	link(u, v);
        link(v, u);
    }

    for (int i=0;i<n;i++) bfs(i);

    for (int epoch=0;epoch<1000;epoch++) {
        int len = strlen(s);
        s[len-8] = '0'+epoch/1000;
        s[len-7] = '0'+epoch/100%10;
        s[len-6] = '0'+epoch/10%10;
        s[len-5] = '0'+epoch%10;
        freopen(s, "w", stdout);
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
                    //printf("%d %d %d\n", u, g, tot);
                    //printf("%d->%d (%d)\n", u, E[u][g], g);
                    u = E[u][g];    
                }

                for (int _=0;_<seq_len;_++) {
                    printf("%d", o[_]-1);
		    if (_!=seq_len-1) printf(", ");
		}

                printf("]\n");
            }
        }
        fclose(stdout);
    }
    fclose(stdin);
    return 0;
}
