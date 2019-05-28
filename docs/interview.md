# Programming Set 2: Let's JOIN

## Overview
This set of problems tests the interviewee's programming competence. Note that this is not a SQL exercise, join's have been chosen to make the understanding of the problem simple. Interviewees are expected to first principles, basic programming language features such as loops, data structures, but not any library or in-built function to do the join (that is the following libraries and functions are not allowed - pandas, plyr, dplyr, dataframe.join(), dataframe.merge()). Because they offer no test of programming competence.  

## Problems
1. Write a function to output the result of a __cross join__ between two lists. 
    * __Input:__ `lt = ['a', 'b', 'c', 'd']; rt = ['e', 'f', 'a', 'a']`
    * __Output:__ `['ae', 'af', 'aa', 'aa', 'be', 'bf', 'ba', 'ba', 'ce', 'cf', 'ca', 'ca', 'de', 'df', 'da', 'da']`
    * __Solution:__ 
        ```python
        def cross_join(lt, rt):
            opt = []
            for l in lt:
                for r in rt:
                    opt.append((l, r))
            return opt
    
        ```
2. Write a function to output the result of a __inner join__ between two lists. 
    * __Input:__ `lt = ['a', 'b', 'c', 'd']; rt = ['e', 'f', 'a', 'a']`
    * __Output:__ : `['aa', 'aa']`
    * __Solution:__
        ```python
        import collections
  
        # Join
        # O(n^2)
        def join(lt, rt):
            opt = []
            for l in lt:
                for r in rt:
                    if l == r:
                        opt.append(l+r)
            return opt
        
        
        # O(m + n)
        def join_on(lt, rt):
            opt = []
            ct = collections.Counter(rt)
            for l in lt:
                if l in ct:
                    for i in range(ct[l]):
                        opt.append(l+l)
            return opt
        ```
3. Write a function to output the result of a __left join__ between two lists.
    * __Input:__ `lt = ['a', 'b', 'c', 'd']; rt = ['e', 'f', 'a', 'a']`
    * __Output:__ `['aa', 'aa', 'b_', 'c_', 'd_']`
    * __Solution:__
        ```python
            # Left Join
            # O(n^2)
            import collections
            def left_join(lt, rt):
                opt = []
                for l in lt:
                    flag = False
                    for r in rt:
                        if l == r:
                            opt.append(l + r)
                            flag = True
                    if not flag:
                        opt.append(l + '_')
                return opt
            
            # O(m + n)
            def left_join_on(lt, rt):
                ct = collections.Counter(rt)
                opt = []
                for l in lt:
                    if l in ct:
                        val = ct[l]
                        for i in range(val):
                            opt.append(l + l)
                    else:
                        opt.append(l + '_')
                return opt
        ```
4. Write a function to output the result of a __full outer join__ between two lists.
    * __Input:__ `lt = ['a', 'b', 'c', 'd']; rt = ['e', 'f', 'a', 'a']`
    * __Output:__ `['aa', 'aa', 'b_', 'c_', 'd_', '_e', '_f']`
    * __Solution:__
        ```python
        # full outer join
        # O(m + n)
        import collections
        def full_outer_join(lt, rt):
            opt = []
            l_ct = collections.Counter(lt)
            r_ct = collections.Counter(rt)
            for l in l_ct:
                if l in r_ct:
                    for i in range(r_ct[l]):
                        opt.append(l+l)
                else:
                    opt.append(l+ '_')
            for r in r_ct:
                if r not in l_ct:
                    opt.append('_' + r)
            return opt

        ```