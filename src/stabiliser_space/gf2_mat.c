/*

    All matrices are 1D arrays
    All addition is performed mod 2

*/

typedef enum { false, true } bool;

int idx(int r, int c, int cs){
    //provides 1D index for 2D array, row-major
    return r * cs + c; 
}

void add_row_to_row(int mat_arr[], int r_add, int r_to, int cs){
    for (int shift = 0; shift < cs; ++shift)
        mat_arr[idx(r_to, shift, cs)] ^= mat_arr[idx(r_add, shift, cs)];
}

void swap_rows(int mat_arr[], int r_0, int r_1, int cs){
    //SWAP = CNOT(0, 1) * CNOT(1, 0) * CNOT(0, 1)
    add_row_to_row(mat_arr, r_1, r_0, cs);
    add_row_to_row(mat_arr, r_0, r_1, cs);
    add_row_to_row(mat_arr, r_1, r_0, cs);
}

void rref(int mat_arr[], int rs, int cs){
    /*
        Places a matrix into reduced-row echelon form using row-swapping
        and adding, where the addition is performed mod 2.
    */

    int s = 0;
    bool cond = false;
    
    for (int x = 0; x < rs; ++x)
    {
        bool b = false;
        while (!b && (x + s < cs))
        {
            if (mat_arr[idx(x, x + s, cs)] == 1) break;
            else if (mat_arr[idx(x, x + s, cs)] == 0)
            {
                for (int y = x; y < rs; ++y)
                {
                    if (mat_arr[idx(y, x + s, cs)] == 1)
                    {
                        swap_rows(mat_arr, y, x, cs);
                        b = true;
                        break;
                    }
                }
            }
            if (!b) s += 1; 
        }
        for (int m = 0; m < rs; ++m)
        {
            cond = (x + s < cs) && (m != x) && (mat_arr[idx(m, x + s, cs)] == 1);
            if (cond) add_row_to_row(mat_arr, x, m, cs);
        }
    }
}

int solve_augmented(int mat_arr[], int rs, int cs, int solution[]){
    /*
        This is a little complicated, let's get into it.
        This is all ultimately copy-paste from Scheinerman's SimpleGF2.
        mat_arr is a 1D array that we're going to have to produce with 
        reshape in numpy at the upper level.
        solution is the intended return value
        This function actually returns a 0 if successful and a 1 if the
        system is inconsistent. 
    */
    bool _in = true;
    // int x = 0;

    int D[rs * cs];
    for (int elem = 0; elem < rs * cs; ++elem) D[elem] = mat_arr[elem];
    rref(D, rs, cs);

    for (int a = 0; a < rs; ++a)
    {
        _in = true;
        for (int b = 0; b < cs - 1; ++b)
        {
            if(D[idx(a, b, cs)] != 0) _in = false;
        }
        if (_in && (D[idx(a, cs - 1, cs)] != 0)) return 1;
    }

    for (int p = 0; p < rs; ++p)
    {
        if (D[idx(p, cs - 1, cs)] == 1)
        {
            for (int n = 0; n < cs - 1; ++n)
            {
                if (D[idx(p, n, cs)] == 1)
                {
                    solution[n] = 1;
                    break;
                }
            }
        }
    }
    return 0;
}