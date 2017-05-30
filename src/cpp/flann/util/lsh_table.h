/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef FLANN_LSH_TABLE_H_
#define FLANN_LSH_TABLE_H_

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits.h>
// TODO as soon as we use C++0x, use the code in USE_UNORDERED_MAP
#if USE_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif
#include <math.h>
#include <stddef.h>
#include <time.h>

#include "flann/util/dynamic_bitset.h"
#include "flann/util/matrix.h"

namespace flann
{

namespace lsh
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** What is stored in an LSH bucket
 */
typedef uint32_t FeatureIndex;
/** The id from which we can get a bucket back in an LSH table
 */
typedef unsigned int BucketKey;

/** A bucket in an LSH table
 */
typedef std::vector<FeatureIndex> Bucket;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** POD for stats about an LSH table
 */
struct LshStats
{
    std::vector<unsigned int> bucket_sizes_;
    size_t n_buckets_;
    size_t bucket_size_mean_;
    size_t bucket_size_median_;
    size_t bucket_size_min_;
    size_t bucket_size_max_;
    size_t bucket_size_std_dev;
    /** Each contained vector contains three value: beginning/end for interval, number of elements in the bin
     */
    std::vector<std::vector<unsigned int> > size_histogram_;
};

/** Overload the << operator for LshStats
 * @param out the streams
 * @param stats the stats to display
 * @return the streams
 */
inline std::ostream& operator <<(std::ostream& out, const LshStats& stats)
{
    size_t w = 20;
    out << "Lsh Table Stats:\n" << std::setw(w) << std::setiosflags(std::ios::right) << "N buckets : "
    << stats.n_buckets_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "mean size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_mean_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "median size : " << stats.bucket_size_median_ << "\n" << std::setw(w)
    << std::setiosflags(std::ios::right) << "min size : " << std::setiosflags(std::ios::left)
    << stats.bucket_size_min_ << "\n" << std::setw(w) << std::setiosflags(std::ios::right) << "max size : "
    << std::setiosflags(std::ios::left) << stats.bucket_size_max_;

    // Display the histogram
    out << std::endl << std::setw(w) << std::setiosflags(std::ios::right) << "histogram : "
    << std::setiosflags(std::ios::left);
    for (std::vector<std::vector<unsigned int> >::const_iterator iterator = stats.size_histogram_.begin(), end =
             stats.size_histogram_.end(); iterator != end; ++iterator) out << (*iterator)[0] << "-" << (*iterator)[1] << ": " << (*iterator)[2] << ",  ";

    return out;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Lsh hash table. As its key is a sub-feature, and as usually
 * the size of it is pretty small, we keep it as a continuous memory array.
 * The value is an index in the corpus of features (we keep it as an unsigned
 * int for pure memory reasons, it could be a size_t)
 */
template<typename ElementType>
class LshTable
{
public:
    /** A container of all the feature indices. Optimized for space
     */
#if USE_UNORDERED_MAP
    typedef std::unordered_map<BucketKey, Bucket> BucketsSpace;
    typedef std::unordered_map<std::vector<int> , Bucket> BucketsSpaceFloat;
#else
    typedef std::map<BucketKey, Bucket> BucketsSpace;
    typedef std::map<std::vector<int> , Bucket> BucketsSpaceFloat;
#endif

    /** A container of all the feature indices. Optimized for speed
     */
    typedef std::vector<Bucket> BucketsSpeed;
    typedef std::vector<Bucket> BucketsSpeedFloat;

    /** Default constructor
     */
    LshTable()
    {
    }

    /** Default constructor
     * Create the mask and allocate the memory
     * @param feature_size is the size of the feature (considered as a ElementType[])
     * @param key_size is the number of bits that are turned on in the feature
     */
    LshTable(unsigned int /*feature_size*/, unsigned int /*key_size*/)
    {
        std::cerr << "@lahTable::LshTable  LSH is not implemented for that type" << std::endl;
        throw;
    }
    
    /** Default constructor
     * Create the mask and allocate the memory
     * @param feature_size is the size of the feature (considered as a ElementType[])
     * @param key_size is the number of bits that are turned on in the feature
     * @param r fraction length
     */
    LshTable(std::vector<std::vector<float> > & as, std::vector<float> & bs, unsigned int /*feature_size*/, unsigned int /*key_size*/, float r)
    {
        std::cerr << "@lahTable::LshTable  LSH is not implemented for that type" << std::endl;
        throw;
    }

    /** Add a feature to the table
     * @param value the value to store for that feature
     * @param feature the feature itself
     */
    void add(unsigned int value, const ElementType* feature)
    {
        // Add the value to the corresponding bucket
        BucketKey key = getKey(feature);

        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
            buckets_speed_[key].push_back(value);
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
            key_bitset_.set(key);
            buckets_space_[key].push_back(value);
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            buckets_space_[key].push_back(value);
            break;
        }
        }
    }
    void addfloat(unsigned int value, std::vector<float> & feature)
    {
      std::cerr << "@lshTable::addfloat  LSH is not implemented for that type" << std::endl;
      throw;
      return ;
    }

    /** Add a set of features to the table
     * @param dataset the values to store
     */
    void add(const std::vector< std::pair<size_t, ElementType*> >& features)
    {
#if USE_UNORDERED_MAP
        buckets_space_.rehash((buckets_space_.size() + features.size()) * 1.2);
#endif
        // Add the features to the table
        for (size_t i = 0; i < features.size(); ++i) {
        	add(features[i].first, features[i].second);
        }
        // Now that the table is full, optimize it for speed/space
        optimize();
    }
    void addfloats(std::vector< std::pair<size_t, std::vector<float> > >& features)
    {
      std::cerr << "@lshTable::addfloats  LSH is not implemented for that type" << std::endl;
      throw;
      return ;
    }

    inline const int getBucketSpaceSizeFloat(void) const{
      int size;
      size = buckets_space_float_.size();
      return size;
    }
    inline const int getBucketSpaceSizeMax(void) const{
      int sizemax;
      sizemax = buckets_space_float_.max_size();
      return sizemax;
    }
    inline const void printBucketSpace(void) const{
      int count = 0;
      int points = 0;
      BucketsSpaceFloat::const_iterator bucket_it, bucket_end = buckets_space_float_.end();
      for(bucket_it=buckets_space_float_.begin();bucket_it!=bucket_end;bucket_it++,count++){
	int bucketsize = 0;
	std::vector<int> key = bucket_it->first;
	std::vector<int>::const_iterator key_it;
	std::cout<< count << ". " ;
	for(key_it=key.begin();key_it!=key.end();key_it++){
	  std::cout<< *key_it << " "; 
	}
	std::cout<< std::endl;

	Bucket bucket = bucket_it->second;
	Bucket::const_iterator buckeit;
	for(buckeit = bucket.begin();buckeit!=bucket.end();buckeit++){
	  std::cout<< *buckeit << " ";
	  bucketsize++;
	  points++;
	}
	std::cout<< std::endl <<"bucket size: " << bucketsize <<std::endl;
      }
	std::cout<< std::endl <<"total size" << points <<std::endl;
     
      return;
    }

    inline const void printdebug(void) const{
	  
      return;
    }

    /** Get a bucket given the key
     * @param key
     * @return
     */
    inline const Bucket* getBucketFromKey(BucketKey key) const
    {
        // Generate other buckets
        switch (speed_level_) {
        case kArray:
            // That means we get the buckets from an array
            return &buckets_speed_[key];
            break;
        case kBitsetHash:
            // That means we can check the bitset for the presence of a key
            if (key_bitset_.test(key)) return &buckets_space_.find(key)->second;
            else return 0;
            break;
        case kHash:
        {
            // That means we have to check for the hash table for the presence of a key
            BucketsSpace::const_iterator bucket_it, bucket_end = buckets_space_.end();
            bucket_it = buckets_space_.find(key);
            // Stop here if that bucket does not exist
            if (bucket_it == bucket_end) return 0;
            else return &bucket_it->second;
            break;
        }
        }
        return 0;
    }
    
    /** Get a bucket given the key
     * @param key
     * @return
     */
    inline const Bucket* getBucketFromKeyFloat(std::vector<int> & key) const
    {
      
      std::cerr << "@lshTable::getBucketFromKeyFloat  LSH is not implemented for that type" << std::endl;
      throw;
      return 0;
    }

    /** Compute the sub-signature of a feature
     */
    size_t getKey(const ElementType* /*feature*/) const
    {
        std::cerr << "@lshTable::getKey  LSH is not implemented for that type" << std::endl;
        throw;
        return 1;
    }
    
    /** Compute the sub-signature of a float-type feature
     */
    std::vector<int> getKeyFloat(std::vector<float> &feature) const
    {
        std::cerr << "@lshTable::getKeyFloat  LSH is not implemented for that type" << std::endl;
        throw;
        return std::vector<int>();
    }

    /** Get statistics about the table
     * @return
     */
    LshStats getStats() const;

private:
    /** defines the speed fo the implementation
     * kArray uses a vector for storing data
     * kBitsetHash uses a hash map but checks for the validity of a key with a bitset
     * kHash uses a hash map only
     */
    enum SpeedLevel
    {
        kArray, kBitsetHash, kHash
    };

    /** Initialize some variables
     */
    void initialize(size_t key_size)
    {
        speed_level_ = kHash;
        key_size_ = key_size;
    }

    /** Optimize the table for speed/space
     */
    void optimize()
    {
        // If we are already using the fast storage, no need to do anything
        if (speed_level_ == kArray) return;

        // Use an array if it will be more than half full
        if (buckets_space_.size() > ((size_t(1) << key_size_) / 2)) {
            speed_level_ = kArray;
            // Fill the array version of it
            buckets_speed_.resize(size_t(1) << key_size_);
            for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket) buckets_speed_[key_bucket->first] = key_bucket->second;

            // Empty the hash table
            buckets_space_.clear();
            return;
        }

        // If the bitset is going to use less than 10% of the RAM of the hash map (at least 1 size_t for the key and two
        // for the vector) or less than 512MB (key_size_ <= 30)
        if (((std::max(buckets_space_.size(), buckets_speed_.size()) * CHAR_BIT * 3 * sizeof(BucketKey)) / 10
             >= size_t(size_t(1) << key_size_)) || (key_size_ <= 32)) {
            speed_level_ = kBitsetHash;
            key_bitset_.resize(size_t(1) << key_size_);
            key_bitset_.reset();
            // Try with the BucketsSpace
            for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket) key_bitset_.set(key_bucket->first);
        }
        else {
            speed_level_ = kHash;
            key_bitset_.clear();
        }
    }

    template<typename Archive>
    void serialize(Archive& ar)
    {
    	int val;
    	if (Archive::is_saving::value) {
    		val = (int)speed_level_;
    	}
    	ar & val;
    	if (Archive::is_loading::value) {
    		speed_level_ = (SpeedLevel) val;
    	}

    	ar & key_size_;
    	ar & mask_;

    	if (speed_level_==kArray) {
    		ar & buckets_speed_;
    	}
    	if (speed_level_==kBitsetHash || speed_level_==kHash) {
    		ar & buckets_space_;
    	}
		if (speed_level_==kBitsetHash) {
			ar & key_bitset_;
		}
    }
    friend struct serialization::access;

    /** The vector of all the buckets if they are held for speed
     */
    BucketsSpeed buckets_speed_;
    BucketsSpeedFloat buckets_speed_float;
    
    /** The hash table of all the buckets in case we cannot use the speed version
     */
    BucketsSpace buckets_space_;
    BucketsSpaceFloat buckets_space_float_;
    
    /** What is used to store the data */
    SpeedLevel speed_level_;

    /** If the subkey is small enough, it will keep track of which subkeys are set through that bitset
     * That is just a speedup so that we don't look in the hash table (which can be mush slower that checking a bitset)
     */
    DynamicBitset key_bitset_;

    /** The size of the sub-signature in bits
     */
    unsigned int key_size_;

    // Members only used for the unsigned char specialization
    /** The mask to apply to a feature to get the hash key
     * Only used in the unsigned char case
     */
    std::vector<size_t> mask_;

    //new vars for float version of LSH
    int k; // how many h in every g
    int d; //vector length, i.e. dimension
    float r;//fraction length
    std::vector<std::vector<float> > as;//vector list, storing k vectors(a)
    std::vector<float> bs;//stroe k real number b
    
  
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialization for unsigned char

template<>
inline LshTable<unsigned char>::LshTable(unsigned int feature_size, unsigned int subsignature_size)
  {
    //std::cout<< "sizeof size_t: " << sizeof(size_t)<<std::endl;
    //std::cout<< "sizeof char: " << sizeof(char)<<std::endl;
    //std::cout<< "bits of char: " << CHAR_BIT <<std::endl;
    initialize(subsignature_size);
    // Allocate the mask
    mask_ = std::vector<size_t>((size_t)ceil((float)(feature_size * sizeof(char)) / (float)sizeof(size_t)), 0);

    // A bit brutal but fast to code
    std::vector<size_t> indices(feature_size * CHAR_BIT);
    for (size_t i = 0; i < feature_size * CHAR_BIT; ++i) indices[i] = i;
    std::random_shuffle(indices.begin(), indices.end());

    // Generate a random set of order of subsignature_size_ bits
    for (unsigned int i = 0; i < key_size_; ++i) {
        size_t index = indices[i];

        // Set that bit in the mask
        size_t divisor = CHAR_BIT * sizeof(size_t);
        size_t idx = index / divisor; //pick the right size_t index
        mask_[idx] |= size_t(1) << (index % divisor); //use modulo to find the bit offset
    }

    // Set to 1 if you want to display the mask for debug
#if 0
    {
        size_t bcount = 0;
        BOOST_FOREACH(size_t mask_block, mask_){
            out << std::setw(sizeof(size_t) * CHAR_BIT / 4) << std::setfill('0') << std::hex << mask_block
                << std::endl;
            bcount += __builtin_popcountll(mask_block);
        }
        out << "bit count : " << std::dec << bcount << std::endl;
        out << "mask size : " << mask_.size() << std::endl;
        return out;
    }
#endif
}

// Specialization for L2<float>
template<>
inline LshTable<float>::LshTable(std::vector<std::vector<float> > & oas, std::vector<float> & obs, unsigned int d, unsigned int k, float r)
  {
    this->k = k;
    this->d = d;
    this->r = r;
    //std::cout << "@lsh_table.h @<float> constructor, r=" << r << std::endl;
    //std::cout << "@lsh_table.h @<float> constructor, this->r=" << this->r << std::endl;

    const std::vector<float> atmp(d,0);
    as.assign(k,atmp);

    std::vector<std::vector<float> >::iterator oasit, asit;
    for(oasit=oas.begin(),asit=as.begin();oasit!=oas.end();oasit++,asit++){
      if(asit==as.end()){
	std::cout << "@lsh_table.h @<float> constructor, warning: parameter oas is longer than as" << std::endl;
	break;
      }
      asit->assign(oasit->begin(),oasit->end());
    }
    bs.assign(obs.begin(),obs.end());
    // randomly generate k vectors a[j] and k real numbers b[j]
    //float r = 8.0;
    //const std::vector<float> atmp(d,0);
    //as.assign(k,atmp);
    //bs.assign(k,0);
    //seed_random(randseed);
    //std::vector<float>::iterator it;
    //std::vector<std::vector<float> >::iterator asit;
    //for(asit=as.begin();asit!=as.end();asit++){
      //std::cout<< "breaker: ";
      //for(it=asit->begin();it!=asit->end();it++){
    //(*it) = float(gaussrand());
	//(*it) = float(uniformrand(0, r));
	//std::cout<< ". ";
    //}
    // std::random_shuffle(asit->begin(),asit->end(),myrandom);
      //std::cout<< std::endl;
    //}
    //for(it=bs.begin();it!=bs.end();it++){
    //  (*it) = (float) rand()/RAND_MAX * (r) ;
    //}//b is in [0,r]
    key_size_ = k;
    
     // Set to 1 if you want to display the hash functions for debug
#if 0
    {
      int flag=300;
      std::vector<float>::iterator it;
      std::vector<std::vector<float> >::iterator asit;
      for(asit=as.begin();flag>=0&&asit!=as.end();asit++,flag--){
	std::cout<< "vector a: ";
	for(it=(*asit).begin();it!=(*asit).end();it++){
	  std::cout<< *it<< " ";
	}
	std::cout<< std::endl<< std::endl;
      }
      std::cout<< "b: ";
      for(it=bs.begin();it!=bs.end();it++){
	std::cout << *it << " " ;
      }
      std::cout << std::endl;
    }
#endif
}


/** Return the Subsignature of a feature
 * @param feature the feature to analyze
 */
template<>
inline size_t LshTable<unsigned char>::getKey(const unsigned char* feature) const
{
    // no need to check if T is dividable by sizeof(size_t) like in the Hamming
    // distance computation as we have a mask
    const size_t* feature_block_ptr = reinterpret_cast<const size_t*> (feature);

    // Figure out the subsignature of the feature
    // Given the feature ABCDEF, and the mask 001011, the output will be
    // 000CEF
    size_t subsignature = 0;
    size_t bit_index = 1;

    for (std::vector<size_t>::const_iterator pmask_block = mask_.begin(); pmask_block != mask_.end(); ++pmask_block) {
        // get the mask and signature blocks
        size_t feature_block = *feature_block_ptr;
        size_t mask_block = *pmask_block;
        while (mask_block) {
            // Get the lowest set bit in the mask block
            size_t lowest_bit = mask_block & (-(ptrdiff_t)mask_block);
            // Add it to the current subsignature if necessary
            subsignature += (feature_block & lowest_bit) ? bit_index : 0;
            // Reset the bit in the mask block
            mask_block ^= lowest_bit;
            // increment the bit index for the subsignature
            bit_index <<= 1;
        }
        // Check the next feature block
        ++feature_block_ptr;
    }
    return subsignature;
}


// Specialization for L2<float>
template<>
inline std::vector<int> LshTable<float>::getKeyFloat(std::vector<float> &feature) const
{
  std::vector<int> key(k,0);
  std::vector<int>::iterator it;
  std::vector<float>::const_iterator bsit;
  std::vector<std::vector<float> >::const_iterator asit;
  double tmpkey = 0;
  std::vector<float>::const_iterator featureit;
  std::vector<float>::const_iterator ait;
  
  for(it=key.begin(), asit=as.begin(), bsit = bs.begin(); it!=key.end() ; ++it,++bsit,++asit){
    if(bsit==bs.end()){
      std::cout<< "error: more real number b required!"<<std::endl;
     break;
    }
    if(asit==as.end()){
     std::cout<< "error: more vector a required!"<<std::endl;
     break;
    }
    tmpkey = 0;
    for(featureit=feature.begin(),ait=asit->begin();ait!=asit->end();ait++,featureit++){
      if(featureit==feature.end()){
	std::cout<< "error: more feature dimension required!"<<std::endl;
	std::cout<< "FYI: dim(feature)=" << feature.size() << ", dim(a)=" << asit->size() <<std::endl;
	break;
      }
      tmpkey+=(*featureit)*(*ait);
    }
    tmpkey+=(*bsit);
    //std::cout<< "tmpkey is: " << tmpkey <<std::endl;
    *it = floor(tmpkey/r);
  }
  
  // Set to 1 if you want to display the key for debug
#if 0
  {
    std::cout << "@getkeyFloat: feature is";
    for(featureit=feature.begin();featureit!=feature.end();featureit++){
      std::cout<< *featureit << " ";
    }
    std::cout <<std::endl ;   
    std::cout << "@getkeyFloat: key is: ";
    for(it=key.begin();it!=key.end();++it){
      std::cout<< *it << " ";
    }
    std::cout <<std::endl ;
  }
#endif

  return key;
}

    /** Add a float feature to the table
     * @param value the value to store for that feature
     * @param feature the feature itself
     */
template<>
void LshTable<float>::addfloat(unsigned int value, std::vector<float> & feature)
{
  // Add the value to the corresponding bucket
  std::vector<int> key = getKeyFloat(feature);
  
  //switch (speed_level_) {
  //case kArray:
    // That means we get the buckets from an array
  // buckets_speed_float_[key].push_back(value);
  // break;
  //case kBitsetHash:
  // // That means we can check the bitset for the presence of a key
    // key_bitset_.set(key);
    //buckets_space_[key].push_back(value);
    //break;
  //case kHash:
  //{
      // That means we have to check for the hash table for the presence of a key
      buckets_space_float_[key].push_back(value);
      //  break;
      // }
      //}
}


    /** Add a set of float features to the table
     * @param dataset the values to store
     */
template<>
void LshTable<float>::addfloats(std::vector< std::pair<size_t, std::vector<float> > >& features)
{
#if USE_UNORDERED_MAP
        buckets_space_.rehash((buckets_space_.size() + features.size()) * 1.2);
#endif
        // Add the features to the table
        for (size_t i = 0; i < features.size(); ++i) {
#if 0
	  std::cout << "@addFloats: feature["<< i<<"].second is ";
	  std::vector<float> featuresingle = features[i].second;
	  std::vector<float>::iterator featureit;
	  for(featureit=featuresingle.begin();featureit!=featuresingle.end();featureit++){
	    std::cout<< *featureit << " ";
	  }
	  std::cout <<std::endl ;
#endif
	  addfloat(features[i].first, features[i].second);
        }
        // Now that the table is full, optimize it for speed/space
        //optimize();
}
    /** Get a bucket given the key
     * @param key
     * @return
     */
template<>
inline const Bucket* LshTable<float>::getBucketFromKeyFloat(std::vector<int> & key) const
{
  
  //we have to check for the hash table for the presence of a key
  BucketsSpaceFloat::const_iterator bucket_it, bucket_end = buckets_space_float_.end();
  bucket_it = buckets_space_float_.find(key);
  // Stop here if that bucket does not exist
  
	  #if 0
	    {
	      if (bucket_it == bucket_end) {
		std::cout << "@lsh_table @getbucketfromKeyFloat:bucket_it == bucket_end, return 0"  <<std::endl ;	      
	      }else{
		std::cout << "@lsh_table @getbucketfromKeyFloat:bucket_it != bucket_end, return found value"  <<std::endl;    
	      }
	    }
	    
	    #endif
  if (bucket_it == bucket_end) return 0;
  else return &bucket_it->second;
}


template<>
inline LshStats LshTable<unsigned char>::getStats() const
{
    LshStats stats;
    stats.bucket_size_mean_ = 0;
    if ((buckets_speed_.empty()) && (buckets_space_.empty())) {
        stats.n_buckets_ = 0;
        stats.bucket_size_median_ = 0;
        stats.bucket_size_min_ = 0;
        stats.bucket_size_max_ = 0;
        return stats;
    }

    if (!buckets_speed_.empty()) {
        for (BucketsSpeed::const_iterator pbucket = buckets_speed_.begin(); pbucket != buckets_speed_.end(); ++pbucket) {
            stats.bucket_sizes_.push_back(pbucket->size());
            stats.bucket_size_mean_ += pbucket->size();
        }
        stats.bucket_size_mean_ /= buckets_speed_.size();
        stats.n_buckets_ = buckets_speed_.size();
    }
    else {
        for (BucketsSpace::const_iterator x = buckets_space_.begin(); x != buckets_space_.end(); ++x) {
            stats.bucket_sizes_.push_back(x->second.size());
            stats.bucket_size_mean_ += x->second.size();
        }
        stats.bucket_size_mean_ /= buckets_space_.size();
        stats.n_buckets_ = buckets_space_.size();
    }

    std::sort(stats.bucket_sizes_.begin(), stats.bucket_sizes_.end());

    //  BOOST_FOREACH(int size, stats.bucket_sizes_)
    //          std::cout << size << " ";
    //  std::cout << std::endl;
    stats.bucket_size_median_ = stats.bucket_sizes_[stats.bucket_sizes_.size() / 2];
    stats.bucket_size_min_ = stats.bucket_sizes_.front();
    stats.bucket_size_max_ = stats.bucket_sizes_.back();

    // TODO compute mean and std
    /*float mean, stddev;
       stats.bucket_size_mean_ = mean;
       stats.bucket_size_std_dev = stddev;*/

    // Include a histogram of the buckets
    unsigned int bin_start = 0;
    unsigned int bin_end = 20;
    bool is_new_bin = true;
    for (std::vector<unsigned int>::iterator iterator = stats.bucket_sizes_.begin(), end = stats.bucket_sizes_.end(); iterator
         != end; )
        if (*iterator < bin_end) {
            if (is_new_bin) {
                stats.size_histogram_.push_back(std::vector<unsigned int>(3, 0));
                stats.size_histogram_.back()[0] = bin_start;
                stats.size_histogram_.back()[1] = bin_end - 1;
                is_new_bin = false;
            }
            ++stats.size_histogram_.back()[2];
            ++iterator;
        }
        else {
            bin_start += 20;
            bin_end += 20;
            is_new_bin = true;
        }

    return stats;
}

// End the two namespaces
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* FLANN_LSH_TABLE_H_ */
