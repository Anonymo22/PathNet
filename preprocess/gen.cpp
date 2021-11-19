#include <bits/stdc++.h>
#define N 100050
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

char file_input[1000], file_output[1000];
const char *default_file_input = "../edge_input/x.in";
const char *default_file_output = "x_40_4_m.txt";
int main(int argc, char *argv[]) {

    if (argc == 1) {
        cerr << "ERROR: The name of dataset is missing." << endl;
        return 0;
    }

    int dataname_len = strlen(argv[1]);
    int default_input_len = strlen(default_file_input);
 
    for (int i=0;i<default_input_len;i++) file_input[i] = default_file_input[i];
    for (int i=0;i<dataname_len;i++) file_input[i+14] = argv[1][i];
    for (int i=0;i<dataname_len;i++) file_output[i] = argv[1][i];
    for (int i=0;i<3;i++) file_input[dataname_len+14+i] = default_file_input[i+15];
    for (int i=0;i<11;i++) file_output[dataname_len+i] = default_file_output[i+1];

    // cout << "File input: " << file_input << endl;
    // cout << "File output: " << file_output << endl;

    freopen(file_input, "r", stdin);
    freopen(file_output, "w",stdout);
    srand(time(0));
    scanf("%d%d",&n, &m);

    cerr << n << endl;


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
