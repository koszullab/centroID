__author__ = 'hervemn'
# coding: utf-8
import os
import shutil
import h5py
import sys
import numpy as np
from progressbar import ProgressBar
import time
from fragment import basic_fragment as B_frag
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


def build_and_filter(base_folder, size_pyramid,factor,min_bin_per_contig,size_chunk,default_level):
    """ build fragments pyramids for multi scale analysis and remove high sparsity fragments"""


    fact_sub_sampling = factor
    ###########################################
    all_pyramid_folder = os.path.join(base_folder, 'pyramids')
    if not(os.path.exists(all_pyramid_folder)):
        os.mkdir(all_pyramid_folder)
    init_pyramid_folder = os.path.join(all_pyramid_folder, 'pyramid_' + str(1) + '_no_thresh')
    if not(os.path.exists(init_pyramid_folder)):
        init_size_pyramid = 1
        build(base_folder, init_size_pyramid, factor, min_bin_per_contig, size_chunk, default_level)
    init_pyramid_folder_level_0 = os.path.join(init_pyramid_folder, "level_0")
    contig_info = os.path.join(init_pyramid_folder_level_0, '0_contig_info.txt')
    fragments_list = os.path.join(init_pyramid_folder_level_0, '0_fragments_list.txt')
    init_abs_fragments_contacts = os.path.join(init_pyramid_folder_level_0, '0_abs_frag_contacts.txt')
    ###########################################

    init_pyramid_file = os.path.join(init_pyramid_folder, "pyramid.hdf5")

    ###########################################

    pyramid_folder = os.path.join(all_pyramid_folder, 'pyramid_' + str(size_pyramid) + '_thresh_auto')
    if not(os.path.exists(pyramid_folder)):
        os.mkdir(pyramid_folder)
    level = 0
    pyramid_level_folder = os.path.join(pyramid_folder, "level_" + str(level))
    if not(os.path.exists(pyramid_level_folder)):
        os.mkdir(pyramid_level_folder)

    current_contig_info = os.path.join(pyramid_level_folder, str(level) + "_contig_info.txt")
    current_frag_list = os.path.join(pyramid_level_folder, str(level) + "_fragments_list.txt")
    current_abs_fragments_contacts = os.path.join(pyramid_level_folder, str(level) + "_abs_frag_contacts.txt")
    if not(os.path.exists(current_contig_info) and os.path.exists(current_frag_list) and os.path.exists(current_abs_fragments_contacts)):
        ###########################################
        print "start filtering"
        pyramid_0 = h5py.File(init_pyramid_file)
        thresh = remove_problematic_fragments(contig_info, fragments_list, init_abs_fragments_contacts,
                                              current_contig_info, current_frag_list,
                                              current_abs_fragments_contacts, pyramid_0)
        pyramid_0.close()
        ###########################################
    else:
        print "filtering already done..."

    hdf5_pyramid_file = os.path.join(pyramid_folder,"pyramid.hdf5")
    pyramid_handle = h5py.File(hdf5_pyramid_file)


    pyramid_level_folder = os.path.join(pyramid_folder,"level_"+str(level))
    level_pyramid = str(level)+"_"
    sub_2_super_frag_index_file = os.path.join(pyramid_level_folder,level_pyramid+"sub_2_super_index_frag.txt")
    for level in xrange(0,size_pyramid):
        pyramid_level_folder = os.path.join(pyramid_folder,"level_"+str(level))
        if not(os.path.exists(pyramid_level_folder)):
            os.mkdir(pyramid_level_folder)
        level_pyramid = str(level)+"_"
        new_contig_list_file = os.path.join(pyramid_level_folder,level_pyramid+"contig_info.txt")
        new_fragments_list_file = os.path.join(pyramid_level_folder,level_pyramid+"fragments_list.txt")
        new_abs_fragments_contacts_file = os.path.join(pyramid_level_folder,level_pyramid+"abs_frag_contacts.txt")

        if level>0:
            if os.path.exists(new_contig_list_file) and os.path.exists(new_fragments_list_file) and os.path.exists(new_abs_fragments_contacts_file) \
            and os.path.exists((sub_2_super_frag_index_file)):
                print "level already built"
                nfrags = file_len(new_fragments_list_file) - 1
            else: # this should never append !!!
                print "writing new_files.."
                nfrags = subsample_data_set(current_contig_info, current_frag_list,fact_sub_sampling,current_abs_fragments_contacts,
                    new_abs_fragments_contacts_file,min_bin_per_contig,
                    new_contig_list_file,new_fragments_list_file,sub_2_super_frag_index_file)
        else:
            if os.path.exists(new_contig_list_file) and os.path.exists(new_fragments_list_file) and os.path.exists(new_abs_fragments_contacts_file):
                print "level already built..."
                nfrags = file_len(new_fragments_list_file) - 1

        try:
            status = pyramid_handle.attrs[str(level)] == "done"
        except KeyError:
            pyramid_handle.attrs[str(level)] = "pending"
            status = False
        if not(status):
            print "Start filling the pyramid"
            level_to_fill = pyramid_handle.create_dataset(str(level),(nfrags,nfrags),'i')
            fill_pyramid_level(level_to_fill,new_abs_fragments_contacts_file, size_chunk,nfrags)
            pyramid_handle.attrs[str(level)] = "done"
            ################################################
        current_frag_list = new_fragments_list_file
        current_contig_info = new_contig_list_file
        current_abs_fragments_contacts = new_abs_fragments_contacts_file
        sub_2_super_frag_index_file = os.path.join(pyramid_level_folder,level_pyramid+"sub_2_super_index_frag.txt")
    print "pyramid built."
    pyramid_handle.close()
    ###############################################
    obj_pyramid = pyramid(pyramid_folder, size_pyramid, default_level,)

    return obj_pyramid



def build( base_folder,size_pyramid, factor, min_bin_per_contig, size_chunk,default_level):
    """ build fragments pyramids for multi scale analysis """
    fact_sub_sampling = factor
    contig_info = os.path.join(base_folder,'info_contigs.txt')
    fragments_list = os.path.join(base_folder,'fragments_list.txt')
    init_abs_fragments_contacts = os.path.join(base_folder,'abs_fragments_contacts_weighted.txt')
    all_pyramid_folder = os.path.join(base_folder,'pyramids')
    pyramid_folder = os.path.join(all_pyramid_folder,'pyramid_'+str(size_pyramid)+'_no_thresh')


    if not(os.path.exists(all_pyramid_folder)):
        os.mkdir(all_pyramid_folder)

    if not(os.path.exists(pyramid_folder)):
        os.mkdir(pyramid_folder)

    hdf5_pyramid_file = os.path.join(pyramid_folder,"pyramid.hdf5")
    pyramid_handle = h5py.File(hdf5_pyramid_file)
    level = 0
    pyramid_level_folder = os.path.join(pyramid_folder,"level_"+str(level))
    if not(os.path.exists(pyramid_level_folder)):
        os.mkdir(pyramid_level_folder)

    current_contig_info = os.path.join(pyramid_level_folder,str(level)+"_contig_info.txt")
    current_frag_list = os.path.join(pyramid_level_folder,str(level)+"_fragments_list.txt")
    current_abs_fragments_contacts = os.path.join(pyramid_level_folder,str(level)+"_abs_frag_contacts.txt")
    for level in xrange(0,size_pyramid):
        pyramid_level_folder = os.path.join(pyramid_folder,"level_"+str(level))
        if not(os.path.exists(pyramid_level_folder)):
            os.mkdir(pyramid_level_folder)
        level_pyramid = str(level)+"_"
        if level == 0:
            shutil.copyfile(contig_info,current_contig_info)
            shutil.copyfile(init_abs_fragments_contacts,current_abs_fragments_contacts)
            nfrags = init_frag_list(fragments_list,current_frag_list)
            new_abs_fragments_contacts_file = current_abs_fragments_contacts
            new_contig_list_file = current_contig_info
            new_fragments_list_file = current_frag_list
            sub_2_super_frag_index_file = os.path.join(pyramid_level_folder,level_pyramid+"sub_2_super_index_frag.txt")

        else:

            new_contig_list_file = os.path.join(pyramid_level_folder,level_pyramid+"contig_info.txt")
            new_fragments_list_file = os.path.join(pyramid_level_folder,level_pyramid+"fragments_list.txt")
            new_abs_fragments_contacts_file = os.path.join(pyramid_level_folder,level_pyramid+"abs_frag_contacts.txt")
            if os.path.exists(new_contig_list_file) and os.path.exists(new_fragments_list_file) and os.path.exists(new_abs_fragments_contacts_file) \
            and os.path.exists(sub_2_super_frag_index_file):
                print "level already built..."
                nfrags = file_len(new_fragments_list_file) - 1
            else:
                print "writing new_files.."
                nfrags = subsample_data_set(current_contig_info, current_frag_list,fact_sub_sampling,current_abs_fragments_contacts,
                    new_abs_fragments_contacts_file,min_bin_per_contig,
                    new_contig_list_file,new_fragments_list_file,sub_2_super_frag_index_file)
        ################################################

        try:
            status = pyramid_handle.attrs[str(level)] == "done"
        except KeyError:
            pyramid_handle.attrs[str(level)] = "pending"
            status = False
        if not(status):
            print "Start filling the pyramid"
            level_to_fill = pyramid_handle.create_dataset(str(level),(nfrags,nfrags),'i')
            fill_pyramid_level(level_to_fill,new_abs_fragments_contacts_file, size_chunk,nfrags)
            pyramid_handle.attrs[str(level)] = "done"
        ################################################
        current_frag_list = new_fragments_list_file
        current_contig_info = new_contig_list_file
        current_abs_fragments_contacts = new_abs_fragments_contacts_file
        sub_2_super_frag_index_file = os.path.join(pyramid_level_folder,level_pyramid+"sub_2_super_index_frag.txt")
    print "pyramid built."
    pyramid_handle.close()
    ###############################################
    # obj_pyramid = pyramid(pyramid_folder,size_pyramid,use_gpu,default_level,use_gpu)
    #
    # return obj_pyramid

def fill_pyramid_level(hdf5_data,abs_contacts_file,size_chunk, nfrags):
    """ fill a pyramid level """
    i = 1
    print "here we go"
    p = ProgressBar('green', width=20, block='▣', empty='□')
    chunk_points = xrange(0,nfrags,size_chunk)
    n_chunk_points = len(chunk_points)
    for t in chunk_points:
        pt = np.float32(i)/n_chunk_points
        p.render(pt * 100, 'step %s\nProcessing...\nDescription: loading numpy chunk into hdf5.' % i)

        limit = min([nfrags - 1,t + size_chunk - 1])
        index_ok = xrange(t,limit + 1,1)
        curr_size_chunk = len(index_ok)
        chunk = np.zeros((curr_size_chunk, nfrags),dtype = np.int32)
        handle_fragments_contacts = open(abs_contacts_file,'r')
        handle_fragments_contacts.readline()

        while 1:
            line_contact = handle_fragments_contacts.readline()
            if not line_contact:
                handle_fragments_contacts.close()
                break

            data = line_contact.split()
            id_abs_a = int(data[0]) - 1
            id_abs_b = int(data[1]) - 1
            if (id_abs_a >=t) and (id_abs_a<= limit):
                chunk[id_abs_a - t,id_abs_b] +=1
            if (id_abs_b >=t) and (id_abs_b<= limit):
                chunk[id_abs_b - t,id_abs_a] +=1

        hdf5_data[index_ok,0:nfrags] = chunk
        i +=1

    print "Done."

def init_frag_list(fragment_list,new_frag_list):
    """ adapt the original frag list to fit the build function requirements """
    handle_frag_list = open(fragment_list,'r')
    handle_new_frag_list = open(new_frag_list,'w')
    handle_new_frag_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%('id', 'chrom', 'start_pos', 'end_pos', 'size',
                                                                       'gc_content', 'accu_frag', 'frag_start', 'frag_end'))
    handle_frag_list.readline()
    i = 0
    while 1:
        line_frag = handle_frag_list.readline()
        if not line_frag:
            handle_frag_list.close()
            handle_new_frag_list.close()
            break
        i += 1
        data = line_frag.split('\t')
        id_init = data[0]
        contig_name = data[1]
        start_pos = data[2]
        end_pos = data[3]
        length_kb = data[4]
        gc_content = str(float(data[5]))
        accu_frag = str(1)
        frag_start = id_init
        frag_end = id_init
        handle_new_frag_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(id_init,contig_name,start_pos,end_pos,length_kb,gc_content,accu_frag,frag_start,frag_end))

    return i


def subsample_data_set( contig_info, fragments_list,fact_sub_sample,abs_fragments_contacts,new_abs_fragments_contacts_file,
                        min_bin_per_contig,
                        new_contig_list_file,new_fragments_list_file, old_2_new_file):

    import numpy as np
    print "fact sub sampling = ", fact_sub_sample
    print "minimum bin numer per contig = ", min_bin_per_contig
    if fact_sub_sample <= 1:
        print "subsampling : nothing to do"
        shutil.copy(fragments_list,new_fragments_list_file)
        shutil.copy(contig_info,new_contig_list_file)
        shutil.copy(abs_fragments_contacts,new_abs_fragments_contacts_file)
        nfrags = file_len(fragments_list) - 1
        handle_old_2_new = open(old_2_new_file,'w')
        handle_old_2_new.write("%s\t%s\n"%("current_id","super_id"))
        for ind in xrange(0,nfrags):
            curr_id = str(ind + 1)
            super_id = curr_id
            handle_old_2_new.write("%s\t%s\n"%(curr_id,super_id))
        handle_old_2_new.close()
    else:
        print "subsampling : start"
        old_2_new_frags = dict()
        spec_new_frags = dict()
        handle_new_contigs_list = open(new_contig_list_file,'w')
        handle_new_contigs_list.write('%s\t%s\t%s\t%s\n' % ('contig','length_kb','n_frags','cumul_length'))

        new_abs_id_frag = 0
        id_frag_abs = 0

        ##### reading contig info !!!! #######################################
        handle_contig_info = open(contig_info,'r')
        handle_contig_info.readline()
        sum_length_contigs = 0
        while 1:
            line_contig = handle_contig_info.readline()

            if not line_contig:
                handle_contig_info.close()
                handle_new_contigs_list.close()
                break

            data = line_contig.split('\t')
            init_contig = data[0]
            id_frag_start = 1
            id_frag_end = int(data[2])
            length_kb = data[1]
            orientation = 'w'
            condition_sub_sample = (id_frag_end / np.float32(fact_sub_sample) ) >= min_bin_per_contig and not (fact_sub_sample ==1)
            accu_frag = 0
            new_rel_id_frag = 0
            id_frag_rel = 0
            sum_length_contigs += id_frag_end
            if condition_sub_sample:
                for arbind in range(0,id_frag_end ):
                #            for id_frag_rel in range(1,id_frag_end+1 ):
                    id_frag_rel += 1
                    id_frag_abs += 1
                    if id_frag_rel%fact_sub_sample == 1:
                        accu_frag = 0
                        new_abs_id_frag += 1
                        new_rel_id_frag += 1
                        spec_new_frags[new_abs_id_frag] = dict()
                        spec_new_frags[new_abs_id_frag]['frag_start'] = id_frag_abs
                        spec_new_frags[new_abs_id_frag]['frag_end'] = id_frag_abs

                    accu_frag += 1
                    old_2_new_frags[id_frag_abs] = new_abs_id_frag
                    spec_new_frags[new_abs_id_frag]['accu_frag'] = accu_frag
                    spec_new_frags[new_abs_id_frag]['id_rel'] = new_rel_id_frag
                    spec_new_frags[new_abs_id_frag]['init_contig'] = init_contig
                    spec_new_frags[new_abs_id_frag]['gc_content'] = []
                    spec_new_frags[new_abs_id_frag]['size'] = []
                    spec_new_frags[new_abs_id_frag]['frag_end'] = id_frag_abs

            else:
                for arbind in xrange(0,id_frag_end ):

                    id_frag_abs += 1
                    new_abs_id_frag += 1
                    new_rel_id_frag += 1
                    id_frag_rel += 1
                    old_2_new_frags[id_frag_abs] =  new_abs_id_frag
                    spec_new_frags[new_abs_id_frag] = {'frag_start':id_frag_abs,'frag_end':id_frag_abs,'accu_frag' : 1 ,
                                                       'init_contig' : init_contig,'gc_content':[],'size':[],'id_rel':new_rel_id_frag}

            handle_new_contigs_list.write('%s\t%s\t%s\t%s\n' % (init_contig,length_kb,new_rel_id_frag,new_abs_id_frag-new_rel_id_frag))
            # write new fragments list
        print "size matrix before sub sampling = ",id_frag_abs
        print "size matrix after sub sampling = ",new_abs_id_frag
        print "sum length contigs = ",sum_length_contigs

        ##### reading fragments list !!!! #######################################
        handle_fragments_list = open(fragments_list,'r')
        handle_fragments_list.readline()
        id_abs = 0
        while 1:
            line_fragments = handle_fragments_list.readline()
            if not line_fragments:
                handle_fragments_list.close()
                break
            id_abs +=  1
            #        print id_abs
            data = line_fragments.split('\t')
            id_init = int(data[0])
            contig_name = data[1]
            start_pos = int(data[2])
            end_pos = int(data[3])
            length_kb = int(data[4])
            gc_content = float(data[5])
            np_id_abs = id_abs
            curr_id = id_init
            init_frag_start = int(data[7])
            init_frag_end = int(data[8])
            id_new = old_2_new_frags[id_abs]
            spec_new_frags[id_new]['gc_content'].append(gc_content)
            #        spec_new_frags[id_new]['size'].append(length_kb)

            if id_abs == spec_new_frags[id_new]['frag_start']:
                spec_new_frags[id_new]['start_pos'] = start_pos
                spec_new_frags[id_new]['init_frag_start'] = init_frag_start # coord level 0
            if id_abs == spec_new_frags[id_new]['frag_end']:
                spec_new_frags[id_new]['end_pos'] = end_pos
                spec_new_frags[id_new]['size'] = end_pos - spec_new_frags[id_new]['start_pos']
                spec_new_frags[id_new]['init_frag_end'] = init_frag_end # coord level 0

        print id_abs
        keys_new_frags = spec_new_frags.keys()
        keys_new_frags.sort()
        handle_new_fragments_list = open(new_fragments_list_file,'w')
        handle_new_fragments_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%('id','chrom','start_pos','end_pos','size',
                                                                                'gc_content','accu_frag','init_frag_start','init_frag_end',
                                                                                'sub_frag_start','sub_frag_end'))
        # print "id problem",spec_new_frags[1]
        nfrags = len(keys_new_frags)
        print "nfrags = ",nfrags
        for new_frag in keys_new_frags:
            id = str(spec_new_frags[new_frag]['id_rel'])

            gc_content = np.array(spec_new_frags[new_frag]['gc_content']).mean()
            size = spec_new_frags[new_frag]['size']

            start_pos = spec_new_frags[new_frag]['start_pos']
            end_pos = spec_new_frags[new_frag]['end_pos']
            chrom = spec_new_frags[new_frag]['init_contig']

            init_frag_start = spec_new_frags[new_frag]['init_frag_start']
            init_frag_end = spec_new_frags[new_frag]['init_frag_end']
            sub_frag_start = spec_new_frags[new_frag]['frag_start']
            sub_frag_end = spec_new_frags[new_frag]['frag_end']
            accu_frag = str( int(init_frag_end) - int(init_frag_start) +1 )
            ##########################
            handle_new_fragments_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(id,chrom,start_pos,end_pos,size,
                                                                                    gc_content,accu_frag,init_frag_start,init_frag_end,
                                                                                    sub_frag_start,sub_frag_end))
            ##########################

        handle_new_fragments_list.close()
        print "new fragments list written..."
        print "..."

        ### be carefull : le dictionnaire est base sur
        if not(abs_fragments_contacts == 'SIMU'):
            print "update contacts files..."
            # write new contacts file
            handle_new_abs_fragments_contacts = open(new_abs_fragments_contacts_file,'w')
            handle_abs_fragments_contacts = open(abs_fragments_contacts,'r')
            handle_new_abs_fragments_contacts.write("%s\t%s\t%s\t%s\t%s\n"%('id_read_a','id_read_b','w_length','w_gc','w_sub_sample'))
            handle_abs_fragments_contacts.readline()
            while 1:
                line_contacts = handle_abs_fragments_contacts.readline()
                if not line_contacts:
                    handle_abs_fragments_contacts.close()
                    handle_new_abs_fragments_contacts.close()
                    break
                data = line_contacts.split()
                w_size = data[2]
                w_gc = data[3]
                abs_id_frag_a = int(data[0])
                abs_id_frag_b = int(data[1])
                new_abs_id_frag_a = old_2_new_frags[abs_id_frag_a]
                new_abs_id_frag_b = old_2_new_frags[abs_id_frag_b]
                w_sub_sample = (spec_new_frags[new_abs_id_frag_a]['accu_frag']* spec_new_frags[new_abs_id_frag_b]['accu_frag'])
                handle_new_abs_fragments_contacts.write("%s\t%s\t%s\t%s\t%s\n"%(str(new_abs_id_frag_a),str(new_abs_id_frag_b),
                                                                                w_size,w_gc,str(w_sub_sample)))
        print("subsampling: done.")
        handle_old_2_new = open(old_2_new_file,'w')
        handle_old_2_new.write("%s\t%s\n"%("current_id","super_id"))
        for ind in old_2_new_frags.keys():
            curr_id = str(ind)
            super_id = str(old_2_new_frags[ind])
            handle_old_2_new.write("%s\t%s\n"%(curr_id,super_id))

        handle_old_2_new.close()

    return nfrags

def remove_problematic_fragments(contig_info, fragments_list,abs_fragments_contacts,new_contig_list_file,
                                 new_fragments_list_file,new_abs_fragments_contacts_file,pyramid):

    import numpy as np

    p = ProgressBar('blue', width=20, block='▣', empty='□')

    full_resolution = pyramid["0"]
    nfrags = full_resolution.shape[0]
    np_nfrags = np.float32(nfrags)
    collect_sparsity = []
    step = 0
    for i in range(0, nfrags):
        v = full_resolution[i, :]
        zeros = v[v==0]
        sparsity = len(zeros)/np_nfrags
        collect_sparsity.append(sparsity)
        step += 1
        pt = step * 100 / nfrags
        p.render(pt, 'step %s\nProcessing...\nDescription: computing sparsity per frag.' % step)
    collect_sparsity = np.array(collect_sparsity,dtype = np.float32)
    mean_spars = collect_sparsity.mean()
    std_spars = collect_sparsity.std()
    max_spars = collect_sparsity.max()

    print "mean sparsity = ", mean_spars
    print "std sparsity = ", std_spars
    print "max_sparsity = ", max_spars

    # thresh = max_spars + std_spars
    thresh = mean_spars + std_spars
    list_fragments_problem = np.nonzero(collect_sparsity>=thresh)[0]

    print "cleaning : start"
    import numpy as np
    print "number of fragments to remove = ", len(list_fragments_problem)

    handle_new_fragments_list = open(new_fragments_list_file,'w')
    handle_new_fragments_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ('id', 'chrom', 'start_pos', 'end_pos',
                                                                              'size', 'gc_content', 'accu_frag',
                                                                              'frag_start', 'frag_end'))

    # build np_id_2_frag dictionary
    np_id_2_frag = get_frag_info_from_fil(fragments_list)
    n_total_frags = len(np_id_2_frag.keys())
    # build Contigs.init_contigs dictionary ( need to know the number of frags per contig)
    init_contigs = get_contig_info_from_file(contig_info)

    prob_frag = dict()
    for np_index in list_fragments_problem:
        tmp_frag = np_id_2_frag[np_index]
        id = tmp_frag['index'] + '-' + tmp_frag['init_contig']
        prob_frag[id] = {'init_contig' : tmp_frag['init_contig'],'index' :tmp_frag['index']}

#    print prob_frag
    contig_info_dict = dict()

    list_init_contig = init_contigs.keys()
    list_init_contig.sort()
    for chrom in list_init_contig:
        contig_info_dict[chrom] ={'n_frags': init_contigs[chrom]['n_frags'],'n_new_frags':0}

    new_id_frag_rel = 0
    new_id_frag_abs = 1
    init_id_frag_abs = 0
    old_2_new_frags = dict()
    handle_fragments_list = open(fragments_list,'r')
    handle_fragments_list.readline()
    spec_new_frags = dict()
    tmp_cumul = {'start_pos':0,'end_pos': 0,'chrom':0,'size':0,'accu_frag':0,'gc_content':[],'lock': False,'init_id_frags':[] ,'list_chrom':[]}
    ######################################
    step = 0
    p = ProgressBar('blue', width=20, block='▣', empty='□')
    while 1:
        line_fragment = handle_fragments_list.readline()
        if not line_fragment:
            if tmp_cumul['lock']:
                for ele in tmp_cumul['init_id_frags']:
                    old_2_new_frags[ele] = 'destroyed'
                new_id_frag_abs -= 1
            handle_fragments_list.close()
            handle_new_fragments_list.close()
            break
        step += 1
        pt = step*100/n_total_frags



        init_id_frag_abs += 1

        data = line_fragment.split('\t')
        id = int(data[0])
        if id == 1:
            new_id_frag_rel = 1
            if not tmp_cumul['lock']:
                new_id_frag_abs += 0
            else:
                for ele in tmp_cumul['init_id_frags']:
                    old_2_new_frags[ele] = 'destroyed'
            tmp_cumul['gc_content'] = []
            tmp_cumul['start_pos'] = 0
            tmp_cumul['init_id_frags'] = []
            tmp_cumul['list_chrom'] = []
            tmp_cumul['frag_start'] = []
            tmp_cumul['frag_end'] = []

        chrom = data[1]
        start_pos = data[2]
        end_pos = data[3]
        size = int(data[4])
        gc_content = float(data[5])
        accu_frag = int(data[6])
        frag_start = int(data[7])
        frag_end = int(data[8])
        name_frag = str(id)+'-'+chrom
        lock = prob_frag.has_key(name_frag)

        tmp_cumul['chrom'] = chrom
        tmp_cumul['list_chrom'].append(chrom)
        tmp_cumul['end_pos'] = end_pos
        tmp_cumul['size'] += size
        tmp_cumul['accu_frag'] += accu_frag
        tmp_cumul['lock'] = prob_frag.has_key(name_frag)
        tmp_cumul['frag_start'].append(frag_start)
        tmp_cumul['frag_end'].append(frag_end)

        tmp_cumul['gc_content'].append(gc_content)
        tmp_cumul['init_id_frags'].append(init_id_frag_abs)

        old_2_new_frags[init_id_frag_abs] = new_id_frag_abs
        if not lock :
            for ele in tmp_cumul['list_chrom']:
                if not(ele == tmp_cumul['list_chrom'][0]):
                    print "warning problem hetero fragments!!!!!!!!!!!!!!!"

            contig_info_dict[chrom]['n_new_frags'] +=1
#            str_frag_start = str(min(tmp_cumul["frag_start"]))
#            str_frag_end = str(max(tmp_cumul["frag_end"]))
            str_frag_start = str(new_id_frag_rel)
            str_frag_end = str(new_id_frag_rel)
            spec_new_frags[new_id_frag_abs] = {'accu_frag':accu_frag,'gc_content':tmp_cumul['gc_content'],'chrom':chrom}
            handle_new_fragments_list.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(str(new_id_frag_rel),chrom,tmp_cumul['start_pos'],
                                                                            tmp_cumul['end_pos'],str(tmp_cumul['size']),
                                                                            str(np.array(tmp_cumul['gc_content']).mean()),tmp_cumul['accu_frag'],str_frag_start,str_frag_end))
            tmp_cumul['start_pos'] = end_pos
            tmp_cumul['end_pos'] = start_pos
            tmp_cumul['size'] = 0
            tmp_cumul['accu_frag'] =0
            tmp_cumul['lock'] = prob_frag.has_key(name_frag)
            tmp_cumul['chrom'] = chrom
            tmp_cumul['gc_content'] = []
            tmp_cumul['init_id_frags'] = []
            tmp_cumul['list_chrom'] = []
            tmp_cumul['frag_start'] = []
            tmp_cumul['frag_end'] = []
            new_id_frag_rel += 1
            new_id_frag_abs += 1
        else:
            p.render(pt ,'step %s\nProcessing...\nDescription: removing bad fragments.' % step)
    p.render(pt ,'step %s\nProcessing...\nDescription: removing bad fragments.' % step)
        ########################################################################################################################
    print 'max new id = ',new_id_frag_abs
    handle_new_contigs_list = open(new_contig_list_file,'w')
    handle_new_contigs_list.write('%s\t%s\t%s\t%s\n' % ('contig','length_kb','n_frags','cumul_length'))

    handle_contig_info = open(contig_info,'r')
    handle_contig_info.readline()
    cumul_length = 0

    while 1:
        line_contig = handle_contig_info.readline()
        if not line_contig:
            handle_contig_info.close()
            handle_new_contigs_list.close()
            break
        data = line_contig.split('\t')
        contig = data[0]
        length_kb = data[1]
        n_frags = contig_info_dict[contig]['n_new_frags']
        if n_frags >0:
            handle_new_contigs_list.write('%s\t%s\t%s\t%s\n' % (contig,str(length_kb),str(n_frags),str(cumul_length)))
            cumul_length += n_frags
        else:
            print contig +' has been deleted...'
    print "update contacts files..."
    # write new contacts file
    n_total_contacts = file_len(abs_fragments_contacts)
    handle_new_abs_fragments_contacts = open(new_abs_fragments_contacts_file,'w')
    handle_abs_fragments_contacts = open(abs_fragments_contacts,'r')



    handle_new_abs_fragments_contacts.write("%s\t%s\t%s\t%s\t%s\n"%('id_read_a','id_read_b','w_length','w_gc','w_sub_sample'))
    handle_abs_fragments_contacts.readline()
    p = ProgressBar('blue', width=20, block='▣', empty='□')
    step = 0
    while 1:
        line_contacts = handle_abs_fragments_contacts.readline()
        if not line_contacts:
            handle_abs_fragments_contacts.close()
            handle_new_abs_fragments_contacts.close()
            break
        data = line_contacts.split()
        w_size = data[2]
        w_gc = data[3]
        abs_id_frag_a = int(data[0])
        abs_id_frag_b = int(data[1])
        new_abs_id_frag_a = old_2_new_frags[abs_id_frag_a]
        new_abs_id_frag_b = old_2_new_frags[abs_id_frag_b]
#        step += 1
#        pt = np.int32(step)/n_total_contacts
#        p.render(pt * 100, 'step %s\nProcessing...\nDescription: updating contacts file.' % step)
        if not(new_abs_id_frag_a == 'destroyed' or new_abs_id_frag_b == 'destroyed'):
            w_sub_sample = (spec_new_frags[new_abs_id_frag_a]['accu_frag']* spec_new_frags[new_abs_id_frag_b]['accu_frag'])
            handle_new_abs_fragments_contacts.write("%s\t%s\t%s\t%s\t%s\n"%(str(new_abs_id_frag_a),str(new_abs_id_frag_b),
                                                                            w_size, w_gc, str(w_sub_sample)))

    return thresh

def get_contig_info_from_file(contig_info):

    handle_contig_info = open(contig_info,'r')
    handle_contig_info.readline()
    init_contigs = dict()
    while 1:
        line_contig = handle_contig_info.readline()
        if not line_contig:
            handle_contig_info.close()
            break

        data = line_contig.split('\t')
        chr = data[0]
        length_kb = int(data[1])
        n_frags = int(data[2])
        cumul_length = int(data[3])
        init_contigs[chr] = dict()
        init_contigs[chr]["n_frags"] = n_frags
        init_contigs[chr]["lemgth_kb"] = length_kb
        init_contigs[chr]["cumul_length"] = cumul_length
    return init_contigs

def get_frag_info_from_fil(fragments_list):
    handle_list_fragments = open(fragments_list,'r')
    handle_list_fragments.readline()
    fragments_info = dict()
    id = 0
    while 1:
        line_contig = handle_list_fragments.readline()
        if not line_contig:
            handle_list_fragments.close()
            break

        data = line_contig.split('\t')
        fragments_info[id] = dict()
        fragments_info[id]["init_contig"] = data[1]
        fragments_info[id]["index"] = data[0]
        id += 1

    return fragments_info




class pyramid():

    def __init__(self,pyramid_folder,n_levels, default_level,):
        print "init pyramid"
        self.pyramid_folder = pyramid_folder
        self.n_levels = n_levels
        pyramid_file = "pyramid.hdf5"
        self.pyramid_file = os.path.join(pyramid_folder,pyramid_file)
        self.data = h5py.File(self.pyramid_file)
        self.spec_level = dict()
        self.default_level = default_level
        self.struct_initiated = False
        self.resol_F_s_kb = 3 # size bin in kb
        self.dist_max_kb = 30 * 2 * self.resol_F_s_kb # length histo in kb
        # self.resol_F_s_kb = 10 # size bin in kb
        # self.dist_max_kb = 10 * 2 * self.resol_F_s_kb # length histo in kb
        for i in xrange(0,n_levels):
            level_folder = os.path.join(pyramid_folder,"level_"+str(i))
            find_super_index = i<n_levels-1
            self.spec_level[str(i)] = dict()
            self.spec_level[str(i)]["level_folder"] = level_folder
            self.spec_level[str(i)]["fragments_list_file"] = os.path.join(level_folder, str(i) + "_fragments_list.txt")
            self.spec_level[str(i)]["contig_info_file"] = os.path.join(level_folder, str(i) + "_contig_info.txt")
            frag_dictionary, contig_dictionary = self.build_frag_dictionnary(self.spec_level[str(i)]["fragments_list_file"],i)
            self.spec_level[str(i)]["fragments_dict"] = frag_dictionary
            self.spec_level[str(i)]["contigs_dict"] = contig_dictionary
            if find_super_index:
                # print "update super index"
                super_index_file = os.path.join(level_folder,str(i)+"_sub_2_super_index_frag.txt")
                self.update_super_index(self.spec_level[str(i)]["fragments_dict"], super_index_file)
                self.update_super_index_in_dict_contig(self.spec_level[str(i)]["fragments_dict"],
                                                       self.spec_level[str(i)]["contigs_dict"])
            else:
                for contig_id in self.spec_level[str(i)]["contigs_dict"].keys():
                    try:
                        t = int(contig_id)
                    except ValueError:
                        self.spec_level[str(i)]["contigs_dict"].pop(contig_id)


        print "object created"

    def close(self):
        self.data.close()

    def build_frag_dictionnary(self, fragments_list, level):
        handle_list_fragments = open(fragments_list,'r')
        handle_list_fragments.readline()
        fragments_info = dict()
        id = 1
        contig_dict = dict()
        id_contig = 0
        while 1:
            line_contig = handle_list_fragments.readline()
            if not line_contig:
                handle_list_fragments.close()
                break

            data = line_contig.split('\t')
            curr_id = int(data[0])
            tag = data[0] + "-" + data[1]
            start_pos = float(data[2])
            end_pos = float(data[3])
            size = float(data[4])
            gc_content = float(data[5])
            id_init_frag_start = data[7]
            id_init_frag_end = data[8]
            if level > 0:
                id_sub_frag_start = data[9]
                id_sub_frag_end = data[10]
            else:
                id_sub_frag_start = curr_id
                id_sub_frag_end = curr_id
            fragments_info[id] = dict()
            contig_name = data[1]
            fragments_info[id]["init_contig"] = contig_name
            fragments_info[id]["index"] = curr_id
            fragments_info[id]["tag"] = tag
            fragments_info[id]["start_pos(bp)"] = start_pos
            fragments_info[id]["end_pos(bp)"] = end_pos
            fragments_info[id]["size(bp)"] = size
            fragments_info[id]["sub_low_index"] = id_sub_frag_start
            fragments_info[id]["sub_high_index"] = id_sub_frag_end
            fragments_info[id]["super_index"] = curr_id
            if not(contig_name in contig_dict):
                id_contig += 1
                contig_dict[contig_name] = dict()
                contig_dict[contig_name]["frag"] = []
                contig_dict[contig_name]["id_contig"] = id_contig
                contig_dict[id_contig] = []
            f = B_frag.initiate(id, curr_id, contig_name, curr_id, start_pos, end_pos, size, gc_content, id_init_frag_start,
                                id_init_frag_end, id_sub_frag_start, id_sub_frag_end, curr_id)
            contig_dict[contig_name]["frag"].append(f)
            contig_dict[id_contig].append(f)
            id += 1
        return fragments_info, contig_dict

    def update_super_index(self,dict_frag, super_index_file):
        handle_super_index = open(super_index_file,'r')
        handle_super_index.readline()
        id = 0
        while 1:
            line_index = handle_super_index.readline()
            if not line_index:
                handle_super_index.close()
                break

            data = line_index.split('\t')
            dict_frag[int(data[0])]["super_index"] = int(data[1])

    def update_super_index_in_dict_contig(self, dict_frag, dict_contig):
        set_contig = set()
        for id in dict_frag.keys():
            id_frag = id
            frag = dict_frag[id_frag]
            init_contig = dict_frag[id_frag]["init_contig"]
            set_contig.add(init_contig)
            id_contig = dict_contig[init_contig]["id_contig"]
            f = dict_contig[id_contig][frag["index"] - 1]
            f.super_index = frag["super_index"]
        # print "set conrigs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # print set_contig
        for ele in set_contig:
            t = dict_contig.pop(ele)

    def zoom_in_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            sub_low = self.spec_level[str_level]["fragments_dict"][frag]["sub_low_index"]
            sub_high = self.spec_level[str_level]["fragments_dict"][frag]["sub_high_index"]
            new_level = level - 1
            for i in range(sub_low, sub_high + 1):
                output.append((i, new_level))
        else:
            output.append(curr_frag)
        return output

    def zoom_out_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            high_frag = self.spec_level[str_level]["fragments_dict"][frag]["super_index"]
            new_level = level + 1
            output = (high_frag, new_level)
        else:
            output = curr_frag
        return output

    def full_zoom_in_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            sub_low = self.spec_level[str_level]["fragments_dict"][frag]["sub_low_index"]
            sub_high = self.spec_level[str_level]["fragments_dict"][frag]["sub_high_index"]
            new_level = level - 1
            for i in range(sub_low, sub_high + 1):
                output.append((i, new_level))
        else:
            output.append(curr_frag)
        return output

    def zoom_in_pixel(self, curr_pixel):
        """ return the curr_frag at a higher resolution"""
        low_frag  = curr_pixel[0]
        high_frag = curr_pixel[1]
        level = curr_pixel[2]
        if level > 0:
            str_level = str(level)
            low_sub_low = self.spec_level[str_level]["fragments_dict"][low_frag]["sub_low_index"]
            low_sub_high = self.spec_level[str_level]["fragments_dict"][low_frag]["sub_high_index"]
            high_sub_low = self.spec_level[str_level]["fragments_dict"][high_frag]["sub_low_index"]
            high_sub_high = self.spec_level[str_level]["fragments_dict"][high_frag]["sub_high_index"]
            vect = [low_sub_low, low_sub_high, high_sub_low, high_sub_high]
            new_pix_low = min(vect)
            new_pix_high = max(vect)
            new_level = level - 1
            new_pixel = [new_pix_low,new_pix_high,new_level]
        else:
            new_pixel = curr_pixel
        return new_pixel


    def zoom_out_pixel(self, curr_pixel):
        """ return the curr_frag at a lower resolution"""
        low_frag  = curr_pixel[0]
        high_frag = curr_pixel[1]
        level = curr_pixel[2]
        str_level = str(level)
        if level<self.n_level - 1:
            low_super = self.spec_level[str_level]["fragments_dict"][low_frag]["super_index"]
            high_super = self.spec_level[str_level]["fragments_dict"][high_frag]["sub_index"]

            new_pix_low = min([low_super,high_super])
            new_pix_high = max([low_super,high_super])
            new_level = level + 1
            new_pixel = [new_pix_low,new_pix_high,new_level]
        else:
            new_pixel = curr_pixel
        return new_pixel

    def zoom_in_area(self, area):
        """ zoom in area"""
        x = area[0]
        y = area[1]
        level = x[2]
        print "x = ", x
        print "y = ", y
        print"level = ",level
        if level == y[2] and level >0:
            new_level = level -1
            high_x = self.zoom_in_pixel(x)
            high_y = self.zoom_in_pixel(y)
            new_x  = [min([high_x[0],high_y[0]]), min([high_x[1],high_y[1]]),new_level]
            new_y  = [max([high_x[0],high_y[0]]), max([high_x[1],high_y[1]]),new_level]
            new_area = [new_x,new_y]
        else:
            new_area = area
        print new_area
        return new_area



