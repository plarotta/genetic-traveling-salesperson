import multiprocessing
import time
import pickle


def adder(arg1, arg2):
    return(arg1+arg2)

def processor(arg1, arg2):
    out = [adder(arg1, arg2), 1, 2, 3, 4]
    print(out)
    
    with open('outfile'+str(arg1), 'wb') as fp:
        pickle.dump(out, fp)




if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(0,10):
        p = multiprocessing.Process(target=processor, args=(i,i+1))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
    print('That took {} seconds'.format(time.time() - starttime))

#to read
    #with open ('outfile', 'rb') as fp: 
        #itemlist = pickle.load(fp)
